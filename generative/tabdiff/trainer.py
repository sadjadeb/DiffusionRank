import os
import glob
import time
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from utils import calculate_metrics
from copy import deepcopy
import math
import random
from tqdm import tqdm
from torch.utils.data import DataLoader


BAR = "=============="
def print_with_bar(log_msg):
    log_msg = BAR + log_msg + BAR
    if "End" in log_msg:
         log_msg += "\n"
    print(log_msg)


class Trainer:
    def __init__(
            self, diffusion, train_data, val_data, idx_val, test_data, idx_test, 
            d_numerical, categories, logger, 
            lr, weight_decay, steps, batch_size, check_val_every,
            sample_batch_size, model_save_path,
            num_samples_to_generate=None,
            lr_scheduler='reduce_lr_on_plateau',
            reduce_lr_patience=100, factor=0.9, 
            ema_decay=0.997,
            closs_weight_schedule = "fixed",
            c_lambda = 1.0,
            d_lambda = 1.0,
            device=torch.device('cuda:1'),
            ckpt_path = None,
            is_finetune=False,
            raw_data_dir=None,
            bell_mu=None,
            bell_sigma=None,
            bell_peak=None,
            approach='pointwise',
            train_data_by_qid=None,
            **kwargs
    ):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()
        self.ema_num_schedule = deepcopy(self.diffusion.num_schedule)
        for param in self.ema_num_schedule.parameters():
            param.detach_()
        self.ema_cat_schedule = deepcopy(self.diffusion.cat_schedule)
        for param in self.ema_cat_schedule.parameters():
            param.detach_()

        self.train_data = train_data
        self.approach = approach
        self.train_data_by_qid = train_data_by_qid
        if approach in ['pairwise', 'listwise_lambdarank'] and train_data_by_qid is not None:
            # For pairwise/listwise_lambdarank, we'll generate pairs dynamically during training
            self.train_iter = None  # Will be generated on-the-fly
        else:
            self.train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_iter = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        self.val_data = val_data
        self.idx_val = idx_val
        self.test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        self.test_data = test_data
        self.idx_test = idx_test
        
        self.d_numerical = d_numerical
        self.categories = categories
        
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.ema_decay = ema_decay
        self.lr_scheduler = lr_scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=factor, patience=reduce_lr_patience)
        self.closs_weight_schedule = closs_weight_schedule
        self.c_lambda = c_lambda
        self.d_lambda = d_lambda

        self.bell_mu = bell_mu
        self.bell_sigma = bell_sigma
        self.bell_peak = bell_peak

        self.batch_size = batch_size
        self.sample_batch_size = sample_batch_size
        self.num_samples_to_generate = num_samples_to_generate
        self.logger = logger
        self.check_val_every = check_val_every
        
        self.device = device
        self.raw_data_dir = raw_data_dir
        self.model_save_path = model_save_path
        self.ckpt_path = ckpt_path
        if self.ckpt_path is not None:
            state_dicts = torch.load(self.ckpt_path, map_location=self.device)
            self.diffusion._denoise_fn.load_state_dict(state_dicts['denoise_fn'])
            self.diffusion.num_schedule.load_state_dict(state_dicts['num_schedule'])
            self.diffusion.cat_schedule.load_state_dict(state_dicts['cat_schedule'])   
            self.optimizer.load_state_dict(state_dicts['optimizer'])
            print(f"Weights are loaded from {self.ckpt_path}")     
        
        if self.ckpt_path is None or is_finetune:
            self.curr_epoch = 0
        else:
            self.curr_epoch = int(os.path.basename(self.ckpt_path).split('_')[-1].split('.')[0])
        
        # Cache for lambda weights (in listwise_lambdarank mode)
        # Structure: {qid: {(idx_i, idx_j): lambda_weight, ...}, ...}
        self.lambda_weights_cache = {}

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
    
    def _generate_pairs(self, batch_size):
        """
        Generate pairs of documents for pairwise training from the same query.
        For each pair, the first document should have a higher label than the second.
        Pairs with equal labels are skipped as they provide no learning signal.
        
        Returns:
            pairs_i: tensor of higher-ranked documents [batch_size, features]
            pairs_j: tensor of lower-ranked documents [batch_size, features]
        """
        pairs_i = []
        pairs_j = []
        
        if self.train_data_by_qid is None:
            raise ValueError("train_data_by_qid must be provided for pairwise training")
        
        # Sample pairs from the same query
        qids = list(self.train_data_by_qid.keys())
        attempts = 0
        max_attempts = batch_size * 10  # Prevent infinite loop
        
        while len(pairs_i) < batch_size and attempts < max_attempts:
            attempts += 1
            qid = random.choice(qids)
            query_data = self.train_data_by_qid[qid]
            n_docs = len(query_data['labels'])
            
            # Need at least 2 documents to form a pair
            if n_docs < 2:
                continue
            
            # Sample two different documents
            idx_i, idx_j = random.sample(range(n_docs), 2)
            label_i = query_data['labels'][idx_i]
            label_j = query_data['labels'][idx_j]
            
            # Skip pairs with equal labels (no preference to learn)
            if label_i == label_j:
                continue
            
            # Ensure document i has higher label than document j
            if label_i < label_j:
                idx_i, idx_j = idx_j, idx_i
            
            pairs_i.append(query_data['features'][idx_i])
            pairs_j.append(query_data['features'][idx_j])
        
        # Convert to tensors
        pairs_i_tensor = torch.from_numpy(np.array(pairs_i)).float().to(self.device)
        pairs_j_tensor = torch.from_numpy(np.array(pairs_j)).float().to(self.device)
        
        return pairs_i_tensor, pairs_j_tensor

    def _compute_lambda_weight_with_scores(self, scores, labels, idx_i, idx_j):
        """
        Compute the lambda weight (|ΔnDCG|) for swapping documents at positions i and j.
        Uses model scores to determine current ranking, then computes NDCG change.
        
        Args:
            scores: array of model scores for all documents in the query
            labels: array of relevance labels for all documents in the query
            idx_i: index of document i (should have higher relevance)
            idx_j: index of document j (should have lower relevance)
        
        Returns:
            Absolute change in NDCG from swapping positions i and j
        """
        # Compute NDCG of current ranking
        current_ndcg = ndcg_score(labels.reshape(1, -1), scores.reshape(1, -1), k=None)
        
        # Create swapped scores by swapping the scores of documents i and j
        swapped_scores = scores.copy()
        swapped_scores[idx_i], swapped_scores[idx_j] = swapped_scores[idx_j], swapped_scores[idx_i]
        
        # Compute NDCG after swap
        swapped_ndcg = ndcg_score(labels.reshape(1, -1), swapped_scores.reshape(1, -1), k=None)
        
        # Return absolute change
        delta_ndcg = abs(current_ndcg - swapped_ndcg)
        
        return delta_ndcg

    def _compute_and_cache_lambda_weights(self):
        """
        Pre-compute lambda weights for all document pairs in all queries.
        This is called once every 50 epochs to populate the lambda_weights_cache.
        """
        self.lambda_weights_cache = {}
        
        self.diffusion.eval()
        print(f"  Pre-computing lambda weights for {len(self.train_data_by_qid)} queries...")
        
        with torch.no_grad():
            for qid, query_data in self.train_data_by_qid.items():
                n_docs = len(query_data['labels'])
                if n_docs < 2:
                    continue
                
                # Get model scores for all documents in this query
                query_features = torch.from_numpy(np.array(query_data['features'])).float().to(self.device)
                query_features_num = query_features[:, :self.d_numerical]
                query_labels = np.array(query_data['labels'])
                
                # Get scores using diffusion model prediction at t=0
                b = query_features_num.shape[0]
                t = torch.zeros(b, device=self.device)
                sigma_num = self.diffusion.num_schedule.total_noise(t[:, None])
                
                # Create masked categorical input
                x_cat_onehot = torch.zeros(b, self.categories[0]+1, device=self.device)
                x_cat_onehot[:, self.categories[0]] = 1.0
                
                _, pred_cat = self.diffusion._denoise_fn(query_features_num, x_cat_onehot, t, sigma=sigma_num)
                query_scores = pred_cat[:, 0].cpu().numpy()
                
                # Initialize cache for this query
                if qid not in self.lambda_weights_cache:
                    self.lambda_weights_cache[qid] = {}
                
                # Compute lambda weights for all pairs with different labels
                for idx_i in range(n_docs):
                    for idx_j in range(idx_i + 1, n_docs):
                        label_i = query_labels[idx_i]
                        label_j = query_labels[idx_j]
                        
                        # Skip pairs with equal labels
                        if label_i == label_j:
                            continue
                        
                        # Ensure document i has higher label than document j
                        if label_i < label_j:
                            actual_idx_i, actual_idx_j = idx_j, idx_i
                        else:
                            actual_idx_i, actual_idx_j = idx_i, idx_j
                        
                        # Compute lambda weight
                        lambda_weight = self._compute_lambda_weight_with_scores(
                            query_scores, query_labels, actual_idx_i, actual_idx_j
                        )
                        lambda_weight = max(lambda_weight, 1e-10)
                        
                        # Store in cache (store both orders for easy lookup)
                        self.lambda_weights_cache[qid][(actual_idx_i, actual_idx_j)] = lambda_weight
                        self.lambda_weights_cache[qid][(actual_idx_j, actual_idx_i)] = lambda_weight
        
        print(f"  Lambda weights cached for {len(self.lambda_weights_cache)} queries")

    def _generate_pairs_with_lambdas(self, batch_size, use_lambda_weights=True):
        """
        Generate pairs of documents with lambda weights for LambdaRank-NDCG training.
        Lambda weights are looked up from the pre-computed cache (no forward passes needed).
        
        Args:
            batch_size: Number of pairs to generate
            use_lambda_weights: If True, use cached NDCG-based lambda weights. 
                               If False, use equal weights (RankNet mode)
        
        Returns:
            pairs_i: tensor of higher-ranked documents [batch_size, features]
            pairs_j: tensor of lower-ranked documents [batch_size, features]
            lambdas: tensor of lambda weights [batch_size]
        """
        if use_lambda_weights and not self.lambda_weights_cache:
            raise ValueError("lambda_weights_cache is empty. Must compute and cache lambda weights first.")
        
        pairs_i = []
        pairs_j = []
        lambdas = []
        
        if self.train_data_by_qid is None:
            raise ValueError("train_data_by_qid must be provided for listwise_lambdarank training")
        
        qids = list(self.train_data_by_qid.keys())
        attempts = 0
        max_attempts = batch_size * 10
        
        while len(pairs_i) < batch_size and attempts < max_attempts:
            attempts += 1
            qid = random.choice(qids)
            query_data = self.train_data_by_qid[qid]
            n_docs = len(query_data['labels'])
            
            if n_docs < 2:
                continue
            
            query_labels = np.array(query_data['labels'])
            
            # Sample two different documents
            idx_i, idx_j = random.sample(range(n_docs), 2)
            label_i = query_labels[idx_i]
            label_j = query_labels[idx_j]
            
            # Skip pairs with equal labels
            if label_i == label_j:
                continue
            
            # Ensure document i has higher label than document j
            if label_i < label_j:
                idx_i, idx_j = idx_j, idx_i
            
            if use_lambda_weights:
                # Look up cached lambda weight (no forward pass needed!)
                if qid in self.lambda_weights_cache and (idx_i, idx_j) in self.lambda_weights_cache[qid]:
                    lambda_weight = self.lambda_weights_cache[qid][(idx_i, idx_j)]
                else:
                    # Fallback to 1.0 if pair not in cache (shouldn't happen for valid pairs)
                    lambda_weight = 1.0
            else:
                # RankNet mode: use equal weights for all pairs
                lambda_weight = 1.0
            
            pairs_i.append(query_data['features'][idx_i])
            pairs_j.append(query_data['features'][idx_j])
            lambdas.append(lambda_weight)
        
        if len(pairs_i) < batch_size:
            print(f"Warning: Only generated {len(pairs_i)} pairs out of {batch_size} requested")
        
        return (torch.from_numpy(np.array(pairs_i)).float().to(self.device), 
                torch.from_numpy(np.array(pairs_j)).float().to(self.device),
                torch.from_numpy(np.array(lambdas)).float().to(self.device))


    def _run_step(self, x, closs_weight, dloss_weight):
        x = x.to(self.device)
        
        self.diffusion.train()
        self.optimizer.zero_grad()

        dloss, closs = self.diffusion.mixed_loss(x)

        if closs_weight == 0.0:
            loss = dloss_weight * dloss + closs_weight * closs.detach()  # detach the continuous loss to avoid backpropagating through it
        elif dloss_weight == 0.0:
            loss = dloss_weight * dloss.detach() + closs_weight * closs  # detach the discrete loss to avoid backpropagating through it
        else:
            loss = dloss_weight * dloss + closs_weight * closs
        
        loss.backward()
        self.optimizer.step()

        return dloss, closs
    
    def _run_step_pairwise(self, x_i, x_j, closs_weight, dloss_weight):
        """
        Pairwise training step using RankNet loss on the categorical label prediction.
        x_i: higher-ranked documents [batch_size, d_numerical + 1] (features + label)
        x_j: lower-ranked documents [batch_size, d_numerical + 1] (features + label)
        """
        x_i = x_i.to(self.device)
        x_j = x_j.to(self.device)
        
        self.diffusion.train()
        self.optimizer.zero_grad()
        
        dloss, closs = self.diffusion.mixed_loss_pairwise(x_i, x_j)
        
        # Combined loss
        if closs_weight == 0.0:
            loss = dloss_weight * dloss + closs_weight * closs.detach()
        elif dloss_weight == 0.0:
            loss = dloss_weight * dloss.detach() + closs_weight * closs
        else:
            loss = dloss_weight * dloss + closs_weight * closs
        
        loss.backward()
        self.optimizer.step()
        
        return dloss, closs

    def _run_step_lambdarank(self, x_i, x_j, lambda_weights, closs_weight, dloss_weight):
        """
        LambdaRank-NDCG training step using RankNet loss weighted by |ΔnDCG|.
        x_i: higher-ranked documents [batch_size, d_numerical + 1] (features + label)
        x_j: lower-ranked documents [batch_size, d_numerical + 1] (features + label)
        lambda_weights: NDCG-based weights for each pair [batch_size]
        """
        x_i = x_i.to(self.device)
        x_j = x_j.to(self.device)
        lambda_weights = lambda_weights.to(self.device)
        
        self.diffusion.train()
        self.optimizer.zero_grad()
        
        dloss, closs = self.diffusion.mixed_loss_lambdarank(x_i, x_j, lambda_weights)
        
        # Combined loss
        if closs_weight == 0.0:
            loss = dloss_weight * dloss + closs_weight * closs.detach()
        elif dloss_weight == 0.0:
            loss = dloss_weight * dloss.detach() + closs_weight * closs
        else:
            loss = dloss_weight * dloss + closs_weight * closs
        
        loss.backward()
        self.optimizer.step()
        
        return dloss, closs

    def compute_ranking_metrics_by_imputation(self, data, idx):
        # Evaluate on validation and test set to compute NDCG and P
        default_num_timesteps = self.diffusion.num_timesteps
        self.diffusion.num_timesteps = 1
        
        x_num = data[:, :self.d_numerical].to(self.device)
        x_cat = data[:, self.d_numerical:].long().to(self.device)
        y_true = data[:, self.d_numerical:].squeeze().cpu().numpy()
        
        if self.approach in ['pairwise', 'listwise_lambdarank']:
            # For pairwise/listwise_lambdarank, get raw logits from model as ranking scores
            # Use minimal noise to get clean predictions
            self.diffusion.eval()
            with torch.no_grad():
                b = x_num.shape[0]
                t = torch.zeros(b, device=self.device)  # t=0 for clean prediction
                sigma_num = self.diffusion.num_schedule.total_noise(t[:, None])
                
                # Create pairs of [0., 0., 1.0] as one-hot encoding for categorical input
                # It means the input is always masked
                x_cat_onehot = torch.zeros(b, self.categories[0]+1, device=self.device)
                x_cat_onehot[:, self.categories[0]] = 1.0
                
                # Get model predictions
                _, pred_cat = self.diffusion._denoise_fn(x_num, x_cat_onehot, t, sigma=sigma_num)
                
                # Use negated class_0 logit as ranking score (same as training)
                # Higher score = more likely to be label=1
                y_pred = pred_cat[:, 0].cpu().numpy()
        else:
            # For pointwise, use sample_impute for discrete prediction
            num_mask_idx, cat_mask_idx = [], [0]
            
            # Set the label column to mask value
            mask_value = self.categories[0] if len(self.categories) > 0 else 2
            x_cat[:, cat_mask_idx] = mask_value
            
            syn_data = self.diffusion.sample_impute(x_num, x_cat, num_mask_idx, cat_mask_idx, 'x_0')
            
            label_column_idx = self.d_numerical
            y_pred = syn_data[:, label_column_idx].cpu().detach().numpy()
        
        results = {}
        for qid, true_label, pred_label in zip(idx, y_true, y_pred):
            if qid not in results:
                results[qid] = []
            results[qid].append((true_label, pred_label))
        
        self.diffusion.num_timesteps = default_num_timesteps
        
        avg_ndcg, avg_map = calculate_metrics(results)
        return avg_ndcg, avg_map
    
    def predict(self, data, idx):
        # Evaluate on validation and test set to compute NDCG and P
        default_num_timesteps = self.diffusion.num_timesteps
        self.diffusion.num_timesteps = 1
        
        x_num = data[:, :self.d_numerical].to(self.device)
        x_cat = data[:, self.d_numerical:].long().to(self.device)
        y_true = data[:, self.d_numerical:].squeeze().cpu().numpy()
        
        if self.approach in ['pairwise', 'listwise_lambdarank']:
            # For pairwise/listwise_lambdarank, get raw logits from model as ranking scores
            # Use minimal noise to get clean predictions
            self.diffusion.eval()
            with torch.no_grad():
                b = x_num.shape[0]
                t = torch.zeros(b, device=self.device)  # t=0 for clean prediction
                sigma_num = self.diffusion.num_schedule.total_noise(t[:, None])
                
                # Create pairs of [0., 0., 1.0] as one-hot encoding for categorical input
                # It means the input is always masked
                x_cat_onehot = torch.zeros(b, self.categories[0]+1, device=self.device)
                x_cat_onehot[:, self.categories[0]] = 1.0
                
                # Get model predictions
                _, pred_cat = self.diffusion._denoise_fn(x_num, x_cat_onehot, t, sigma=sigma_num)
                
                # Use negated class_0 logit as ranking score (same as training)
                # Higher score = more likely to be label=1
                y_pred = pred_cat[:, 0].cpu().numpy()
        else:
            # For pointwise, use sample_impute for discrete prediction
            num_mask_idx, cat_mask_idx = [], [0]
            
            # Set the label column to mask value
            mask_value = self.categories[0] if len(self.categories) > 0 else 2
            x_cat[:, cat_mask_idx] = mask_value
            
            syn_data = self.diffusion.sample_impute(x_num, x_cat, num_mask_idx, cat_mask_idx, 'x_0')
            
            label_column_idx = self.d_numerical
            y_pred = syn_data[:, label_column_idx].cpu().detach().numpy()
        
        results = {}
        for qid, true_label, pred_label in zip(idx, y_true, y_pred):
            if qid not in results:
                results[qid] = []
            results[qid].append((true_label, pred_label))
        
        self.diffusion.num_timesteps = default_num_timesteps
        
        return results

    def run_loop(self):
        closs_weight, dloss_weight = self.c_lambda, self.d_lambda
        best_val_loss = np.inf #noqa: F841
        best_val_ndcg = -np.inf #noqa: F841
        start_time = time.time()
        print_with_bar(f"Starting Training Loop, total number of epoch = {self.steps}")
        # Set up wandb's step metric
        self.logger.define_metric("epoch")
        self.logger.define_metric("*", step_metric="epoch")
        
        start_epoch = self.curr_epoch
        if start_epoch > 0:
            print_with_bar(f"Resuming training from epoch {start_epoch}, with validation check every {self.check_val_every} epoches")
        for epoch in range(start_epoch, self.steps):
            self.curr_epoch = epoch+1
            
            # Compute the loss weights
            if self.closs_weight_schedule == "fixed":
                pass
            elif self.closs_weight_schedule == "anneal":
                frac_done = epoch / self.steps
                closs_weight = self.c_lambda * (1 - frac_done)
            elif self.closs_weight_schedule == "ramp_up":
                frac_done = epoch / self.steps
                closs_weight = self.c_lambda * frac_done
            elif self.closs_weight_schedule == "half_cutoff":
                frac_done = epoch / self.steps
                closs_weight = self.c_lambda if frac_done < 0.5 else 0.0
            elif self.closs_weight_schedule == "triangle":
                frac_done = epoch / self.steps
                if frac_done <= 0.5:
                    closs_weight = self.c_lambda * (frac_done / 0.5)        # ramp up
                else:
                    closs_weight = self.c_lambda * (1 - (frac_done - 0.5)/0.5)  # ramp down
            elif self.closs_weight_schedule == "smooth_cutoff":
                # closs = c_lambda * (1 - sigmoid(k*(x - t0)))
                frac_done = epoch / self.steps
                k = 12.0          # sharpness (same default as plot)
                t0 = 0.5          # center of the transition (same default as plot)
                a = max(min(k * (frac_done - t0), 50.0), -50.0)  # clip to avoid overflow
                sig = 1.0 / (1.0 + math.exp(-a))
                closs_weight = self.c_lambda * (1.0 - sig)
            elif self.closs_weight_schedule == "bell_like":
                # closs_weight = c_lambda * exp(-0.5 * ((x - mu)/sigma)^2) * peak
                # Defaults: peak at halfway (mu=0.5) with moderate width (sigma=0.15) and peak value of 1.0
                frac_done = epoch / self.steps
                mu = self.bell_mu or 0.5
                sigma = self.bell_sigma or 0.15
                bell_peak = self.bell_peak or 1.0

                z = (frac_done - mu) / sigma
                shape = math.exp(-0.5 * z * z)
                closs_weight = self.c_lambda * bell_peak * shape
            else:
                raise NotImplementedError(f"The continuous loss weight schedule {self.closs_weight_schedule} is not implemented")

            # Training Step
            curr_dloss = 0.0
            curr_closs = 0.0
            curr_count = 0
            curr_lr = self.optimizer.param_groups[0]['lr']
            
            if self.approach == 'pairwise' and self.train_data_by_qid is not None:
                # For pairwise, generate pairs dynamically and use pairwise loss
                num_batches = max(1, len(self.train_data) // self.batch_size)
                pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{self.steps}")
                for _ in pbar:
                    x_i, x_j = self._generate_pairs(self.batch_size)
                    batch_dloss, batch_closs = self._run_step_pairwise(x_i, x_j, closs_weight, dloss_weight)
                    curr_dloss += batch_dloss.item() * self.batch_size
                    curr_closs += batch_closs.item() * self.batch_size
                    curr_count += self.batch_size
                    pbar.set_postfix({
                        "lr": curr_lr,
                        "DLoss (RankNet)": np.around(curr_dloss/curr_count, 4),
                        "CLoss": np.around(curr_closs/curr_count, 4),
                        "TotalLoss": np.around((curr_dloss * dloss_weight + curr_closs * closs_weight)/curr_count, 4),
                        "closs_weight": closs_weight,
                        "dloss_weight": dloss_weight,
                    })
            elif self.approach == 'listwise_lambdarank' and self.train_data_by_qid is not None:
                # For listwise_lambdarank, generate pairs with lambda weights and use LambdaRank loss
                epoch_1indexed = epoch + 1  # Convert to 1-indexed for strategy logic
                
                if epoch_1indexed <= 50:
                    use_lambda_weights = False
                    if epoch_1indexed == 1:
                        print("  Using RankNet mode (equal weights) for first 50 epochs")
                else:
                    # Check if we need to recompute lambda weights (at epochs 51, 101, 151, ...)
                    if (epoch_1indexed - 51) % 50 == 0:  # epochs 51, 101, 151, ...
                        print(f"  Epoch {epoch_1indexed}: Computing and caching lambda weights (will be used for epochs {epoch_1indexed}-{epoch_1indexed+49})")
                        self._compute_and_cache_lambda_weights()
                    
                    use_lambda_weights = True
                
                num_batches = max(1, len(self.train_data) // self.batch_size)
                pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{self.steps}")
                for _ in pbar:
                    x_i, x_j, lambda_weights = self._generate_pairs_with_lambdas(self.batch_size, use_lambda_weights=use_lambda_weights)
                    batch_dloss, batch_closs = self._run_step_lambdarank(x_i, x_j, lambda_weights, closs_weight, dloss_weight)
                    curr_dloss += batch_dloss.item() * self.batch_size
                    curr_closs += batch_closs.item() * self.batch_size
                    curr_count += self.batch_size
                    loss_name = "DLoss (LambdaRank)" if use_lambda_weights else "DLoss (RankNet)"
                    pbar.set_postfix({
                        "lr": curr_lr,
                        loss_name: np.around(curr_dloss/curr_count, 4),
                        "CLoss": np.around(curr_closs/curr_count, 4),
                        "TotalLoss": np.around((curr_dloss * dloss_weight + curr_closs * closs_weight)/curr_count, 4),
                        "closs_weight": closs_weight,
                        "dloss_weight": dloss_weight,
                    })
            else:
                pbar = tqdm(self.train_iter, total=len(self.train_iter), desc=f"Epoch {epoch+1}/{self.steps}")
                for batch in pbar:
                    x = batch.float().to(self.device)
                    batch_dloss, batch_closs = self._run_step(x, closs_weight, dloss_weight)
                    curr_dloss += batch_dloss.item() * len(x)
                    curr_closs += batch_closs.item() * len(x)
                    curr_count += len(x)
                    pbar.set_postfix({
                        "lr": curr_lr,
                        "DLoss": np.around(curr_dloss/curr_count, 4),
                        "CLoss": np.around(curr_closs/curr_count, 4),
                        "TotalLoss": np.around((curr_dloss * dloss_weight + curr_closs * closs_weight)/curr_count, 4),
                        "closs_weight": closs_weight,
                        "dloss_weight": dloss_weight,
                    })
            
            # Log Losses
            log_dict = {}
            mloss = np.around(curr_dloss / curr_count, 4)
            gloss = np.around(curr_closs / curr_count, 4)
            total_loss = mloss * dloss_weight + gloss * closs_weight
            
            # if self.approach == 'pairwise':
            #     val_mloss, val_gloss, val_loss = None, None, None
            #     test_mloss, test_gloss, test_loss = None, None, None
            # else:
            #     val_mloss, val_gloss = self.compute_loss_for_eval(self.val_iter)
            #     val_loss = val_mloss + val_gloss
            #     test_mloss, test_gloss = self.compute_loss_for_eval(self.test_iter)
            #     test_loss = test_mloss + test_gloss
            val_mloss, val_gloss, val_loss = None, None, None
            test_mloss, test_gloss, test_loss = None, None, None

            val_ndcg, val_map = self.compute_ranking_metrics_by_imputation(self.val_data, self.idx_val)
            test_ndcg, test_map = self.compute_ranking_metrics_by_imputation(self.test_data, self.idx_test)

            loss_dict = {
                "epoch": epoch + 1,
                "lr": curr_lr,
                "closs_weight": closs_weight,
                "dloss_weight": dloss_weight,
                "loss/train_c_loss": gloss,
                "loss/train_d_loss": mloss,
                "loss/train_total_loss": total_loss,
                "loss/val_c_loss": val_gloss,
                "loss/val_d_loss": val_mloss,
                "loss/val_total_loss": val_loss,
                "ranking_metrics/val_ndcg": val_ndcg,
                "ranking_metrics/val_map": val_map,
                "loss/test_c_loss": test_gloss,
                "loss/test_d_loss": test_mloss,
                "loss/test_total_loss": test_loss,
                "ranking_metrics/test_ndcg": test_ndcg,
                "ranking_metrics/test_map": test_map,
            }
            log_dict.update(loss_dict)
            
            # Log the learned noise schedules for numerical dimensions
            num_noise_dict = {}
            if self.diffusion.num_schedule.rho().dim() > 0 and len(self.diffusion.num_schedule.rho()) > 1:
                num_noise_dict = {f"num_noise/rho_col_{i}": value.item() for i, value in enumerate(self.diffusion.num_schedule.rho())}
            else:
                num_noise_dict = {"num_noise/rho": self.diffusion.num_schedule.rho().item()}
            log_dict.update(num_noise_dict)

            # Log the learned noise schedules for categorical dimensions
            cat_noise_dict = {}
            if self.diffusion.cat_schedule.k().dim() == 0:   # non-learnable cat schedule
                cat_noise_dict = {"cat_noise/k": self.diffusion.cat_schedule.k().item()}
                log_dict.update(cat_noise_dict)
            else:
                if len(self.diffusion.cat_schedule.k()) > 0:    # if categorical data is not empty
                    cat_noise_dict = {f"cat_noise/k_col_{i}": value.item() for i, value in enumerate(self.diffusion.cat_schedule.k())}
                    log_dict.update(cat_noise_dict)
            
            # Adjust learning rate
            if self.lr_scheduler == 'reduce_lr_on_plateau':
                self.scheduler.step(total_loss)
            elif  self.lr_scheduler == 'anneal':
                self._anneal_lr(epoch)
            elif self.lr_scheduler == 'fixed':
                pass
            else:
                raise NotImplementedError(f"LR scheduler with name '{self.lr_scheduler}' is not implemented")
            
            # Save ckpt base on the best training loss
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                to_remove = glob.glob(os.path.join(self.model_save_path, "best_model_*"))
                if to_remove:
                    os.remove(to_remove[0])
                state_dicts = {
                    'denoise_fn': self.diffusion._denoise_fn.state_dict(), 
                    'num_schedule':self.diffusion.num_schedule.state_dict(), 
                    'cat_schedule': self.diffusion.cat_schedule.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }
                torch.save(state_dicts, os.path.join(self.model_save_path, f'best_model_{np.round(val_ndcg,4)}_{epoch+1}.pt'))
            
            # Submit logs
            self.logger.log(log_dict)

        end_time = time.time()
        print_with_bar(f"Ending Trainnig Loop, totoal training time = {end_time - start_time}")
        self.logger.log({'training_time': end_time - start_time})
        
    def to_ema_model(self):
        curr_model = self.diffusion._denoise_fn
        curr_num_schedule = self.diffusion.num_schedule
        curr_cat_schedule = self.diffusion.cat_schedule
        self.diffusion._denoise_fn = self.ema_model  # temporarily install the ema parameters into the model
        self.diffusion.num_schedule = self.ema_num_schedule
        self.diffusion.cat_schedule = self.ema_cat_schedule
        
        return curr_model, curr_num_schedule, curr_cat_schedule

    def to_model(self, curr_model, curr_num_schedule, curr_cat_schedule):
        self.diffusion._denoise_fn = curr_model      # give back the parameters
        self.diffusion.num_schedule = curr_num_schedule
        self.diffusion.cat_schedule = curr_cat_schedule
        
    def test_impute(self, impute_condition, imputed_sample_save_dir):
        self.diffusion.eval()
        
        task_type = "binclass"
        d_numerical, categories = self.d_numerical, self.categories
        num_mask_idx, cat_mask_idx = [], []
        X_test = self.test_data.to(self.device)
        x_num_test, x_cat_test = X_test[:, :d_numerical], X_test[:, d_numerical:].long()
        
        if task_type == 'binclass':    # for cat cols, push the masked col to [MASK]
            cat_mask_idx += [0]
        else:      # for num cols, set the masked col to the col mean
            num_mask_idx += [0]
            avg = x_num_test[:, num_mask_idx].mean(0).to(self.device)
        
        with torch.no_grad():            
            # Apply mask to x_0
            if num_mask_idx:
                x_num_test[:, num_mask_idx] = avg
            if cat_mask_idx:
                x_cat_test[:, cat_mask_idx] = torch.tensor(categories, dtype=x_cat_test.dtype, device=x_cat_test.device)[cat_mask_idx]
            
            # Sample imputed tables
            syn_data = self.diffusion.sample_impute(x_num_test, x_cat_test, num_mask_idx, cat_mask_idx, impute_condition)
            print(f"Shape of the imputed sample = {syn_data.shape}")
            
            label_column_idx = self.d_numerical
            y_pred_test = syn_data[:, label_column_idx]
            y_true_test = self.test_data[:, label_column_idx]
            
            # Save results
            os.makedirs(imputed_sample_save_dir, exist_ok=True)
            results_save_path = os.path.join(imputed_sample_save_dir, 'impute_output.txt')
            with open(results_save_path, 'w') as f:
                f.write('qid true_label pred_label\n')
                for qid, true_label, pred_label in zip(self.idx_test, y_true_test, y_pred_test):
                    f.write(f'{qid} {true_label} {pred_label}\n')
            print(f"Results saved to {results_save_path}")
