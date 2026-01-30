import torch.nn.functional as F
import torch
import math
import numpy as np
from tabdiff.models.noise_schedule import *
from tqdm import tqdm
from itertools import chain

"""
“Our implementation of the continuous-time masked diffusion is inspired by https://arxiv.org/abs/2406.07524's implementation at [https://github.com/kuleshov-group/mdlm], with modifications to support data distributions that include categorical dimensions of different sizes.”
"""

S_churn= 1
S_min=0
S_max=float('inf')
S_noise=1

class UnifiedCtimeDiffusion(torch.nn.Module):
    def __init__(
            self,
            num_classes: np.array,
            num_numerical_features: int,
            denoise_fn,
            num_timesteps=1000,
            scheduler='power_mean',
            cat_scheduler='log_linear',
            noise_dist='uniform',
            edm_params={},
            noise_dist_params={},
            noise_schedule_params={},
            sampler_params={},
            device=torch.device('cpu'),
            **kwargs
        ):

        super(UnifiedCtimeDiffusion, self).__init__()

        self.num_numerical_features = num_numerical_features
        self.num_classes = num_classes # it as a vector [K1, K2, ..., Km]
        self.num_classes_expanded = torch.from_numpy(
            np.concatenate([num_classes[i].repeat(num_classes[i]) for i in range(len(num_classes))])
        ).to(device) if len(num_classes)>0 else torch.tensor([]).to(device).int()
        self.mask_index = torch.tensor(self.num_classes).long().to(device)
        self.neg_infinity = -1000000.0 
        self.num_classes_w_mask = tuple(self.num_classes + 1)

        offsets = np.cumsum(self.num_classes)
        offsets = np.append([0], offsets)
        self.slices_for_classes = []
        for i in range(1, len(offsets)):
            self.slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
        self.offsets = torch.from_numpy(offsets).to(device)
        
        offsets = np.cumsum(self.num_classes) + np.arange(1, len(self.num_classes)+1)
        offsets = np.append([0], offsets)
        self.slices_for_classes_with_mask = []
        for i in range(1, len(offsets)):
            self.slices_for_classes_with_mask.append(np.arange(offsets[i - 1], offsets[i]))

        self._denoise_fn = denoise_fn
        self.num_timesteps = num_timesteps
        self.scheduler = scheduler
        self.cat_scheduler = cat_scheduler
        self.noise_dist = noise_dist
        self.edm_params = edm_params
        self.noise_dist_params = noise_dist_params
        self.sampler_params = sampler_params
        
        self.num_mask_idx = []
        self.cat_mask_idx = []
        
        self.device = device
        
        if self.scheduler == 'power_mean':
            self.num_schedule = PowerMeanNoise(**noise_schedule_params)
        elif self.scheduler == 'power_mean_per_column':
            self.num_schedule = PowerMeanNoise_PerColumn(num_numerical = num_numerical_features, **noise_schedule_params)
        elif self.scheduler == 'linear':
            self.num_schedule = LinearNoise(**noise_schedule_params)
        elif self.scheduler == 'cosine':
            self.num_schedule = CosineNoise(**noise_schedule_params)
        else:
            raise NotImplementedError(f"The noise schedule--{self.scheduler}-- is not implemented for contiuous data at CTIME ")
        
        if self.cat_scheduler == 'log_linear':
            self.cat_schedule = LogLinearNoise(**noise_schedule_params)
        elif self.cat_scheduler == 'log_linear_per_column':
            self.cat_schedule = LogLinearNoise_PerColumn(num_categories = len(num_classes), **noise_schedule_params)
        else:
            raise NotImplementedError(f"The noise schedule--{self.cat_scheduler}-- is not implemented for discrete data at CTIME ")

    def mixed_loss(self, x):
        b = x.shape[0]
        device = x.device

        x_num = x[:, :self.num_numerical_features]
        x_cat = x[:, self.num_numerical_features:].long()

        # Sample noise level
        t = torch.rand(b, device=device, dtype=x_num.dtype)
        t = t[:, None]
        sigma_num = self.num_schedule.total_noise(t)
        sigma_cat = self.cat_schedule.total_noise(t)
        
        # Convert sigma_cat to the corresponding alpha and move_chance
        alpha = torch.exp(-sigma_cat)
        move_chance = 1 - alpha
            
        # Continuous forward diff
        x_num_t = x_num
        if x_num.shape[1] > 0:
            noise = torch.randn_like(x_num)
            x_num_t = x_num + noise * sigma_num
        
        # Discrete forward diff
        x_cat_t = x_cat
        x_cat_t_soft = x_cat # in the case where x_cat is empty, x_cat_t_soft will have the same shape as x_cat
        if x_cat.shape[1] > 0:
            is_learnable = self.cat_scheduler == 'log_linear_per_column'
            strategy = 'soft' if is_learnable else 'hard'
            x_cat_t, x_cat_t_soft = self.q_xt(x_cat, move_chance, strategy=strategy)

        # Predict original data (distribution)
        model_out_num, model_out_cat = self._denoise_fn(   
            x_num_t, x_cat_t_soft,
            t.squeeze(), sigma=sigma_num
        )

        d_loss = torch.zeros((1,)).float()
        c_loss = torch.zeros((1,)).float()

        if x_num.shape[1] > 0:
            c_loss = self._edm_loss(model_out_num, x_num, sigma_num)
        if x_cat.shape[1] > 0:
            logits = self._subs_parameterization(model_out_cat, x_cat_t)    # log normalized probabilities, with the entry mask category being set to -inf
            d_loss = self._absorbed_closs(logits, x_cat, sigma_cat)
            
        return d_loss.mean(), c_loss.mean()
    
    def mixed_loss_pairwise(self, x_i, x_j):
        b = x_i.shape[0]
        device = x_i.device
        
        x_num_i = x_i[:, :self.num_numerical_features]
        x_cat_i = x_i[:, self.num_numerical_features:].long()
        x_num_j = x_j[:, :self.num_numerical_features]
        x_cat_j = x_j[:, self.num_numerical_features:].long()

        # Sample noise level
        t = torch.rand(b, device=device, dtype=x_num_i.dtype)
        t = t[:, None]
        sigma_num = self.num_schedule.total_noise(t)
        sigma_cat = self.cat_schedule.total_noise(t)
        
        # Convert sigma_cat to the corresponding alpha and move_chance
        alpha = torch.exp(-sigma_cat)
        move_chance = 1 - alpha
        
        # Continuous forward diff
        x_num_i_t = x_num_i
        x_num_j_t = x_num_j
        if x_num_i.shape[1] > 0:
            noise_i = torch.randn_like(x_num_i)
            noise_j = torch.randn_like(x_num_j)
            x_num_i_t = x_num_i + noise_i * sigma_num
            x_num_j_t = x_num_j + noise_j * sigma_num
        
        # Discrete forward diff
        # For pairwise: x_t is mask indicator (same for both), x_i/j_t_soft is [label, mask_indicator]
        x_cat_i_t_soft = x_cat_i.float()
        x_cat_j_t_soft = x_cat_j.float()
        if x_cat_i.shape[1] > 0:
            # Returns: x_t (mask indicator), x_i_t_soft [label, mask], x_j_t_soft [label, mask]
            x_t, x_cat_i_t_soft, x_cat_j_t_soft = self.q_xt_pairwise(x_cat_i, x_cat_j, move_chance)
        
        # Predict original data (distribution)
        model_out_num_i, model_out_cat_i = self._denoise_fn(
            x_num_i_t, x_cat_i_t_soft,
            t.squeeze(), sigma=sigma_num
        )
        model_out_num_j, model_out_cat_j = self._denoise_fn(   
            x_num_j_t, x_cat_j_t_soft,
            t.squeeze(), sigma=sigma_num
        )
        
        d_loss = torch.zeros((1,)).float()
        c_loss = torch.zeros((1,)).float()
        
        if x_num_i.shape[1] > 0:
            c_loss_i = self._edm_loss(model_out_num_i, x_num_i, sigma_num)
            c_loss_j = self._edm_loss(model_out_num_j, x_num_j, sigma_num)
            c_loss = (c_loss_i + c_loss_j) / 2
        if x_cat_i.shape[1] > 0:
            score_pred_i = model_out_cat_i[:, 0]  # Higher score for label=1 docs
            score_pred_j = model_out_cat_j[:, 0]  # Lower score for label=0 docs
            d_loss = self._ranknet_loss(score_pred_i, score_pred_j, x_t.squeeze(), sigma_cat)
            
        return d_loss.mean(), c_loss.mean()

    @torch.no_grad()
    def sample(self, num_samples):
        b = num_samples
        device = self.device
        dtype = torch.float32
        
        # Create the chain of t
        t = torch.linspace(0,1,self.num_timesteps, dtype=dtype, device=device)      # times = 0.0,...,1.0
        t = t[:, None]
        
        # Compute the chains of sigma
        sigma_num_cur = self.num_schedule.total_noise(t)
        sigma_cat_cur = self.cat_schedule.total_noise(t)
        sigma_num_next = torch.zeros_like(sigma_num_cur)
        sigma_num_next[1:] = sigma_num_cur[0:-1]
        sigma_cat_next = torch.zeros_like(sigma_cat_cur)
        sigma_cat_next[1:] = sigma_cat_cur[0:-1]
        
        # Prepare sigma_hat for stochastic sampling mode
        if self.sampler_params['stochastic_sampler']:
            gamma = min(S_churn / self.num_timesteps, np.sqrt(2) - 1) * (S_min <= sigma_num_cur) * (sigma_num_cur <= S_max)
            sigma_num_hat = sigma_num_cur + gamma * sigma_num_cur
            t_hat = self.num_schedule.inverse_to_t(sigma_num_hat)
            t_hat = torch.min(t_hat, dim=-1, keepdim=True).values    # take the samllest t_hat induced by sigma_num
            zero_gamma = (gamma==0).any()
            t_hat[zero_gamma] = t[zero_gamma]
            out_of_bound = (t_hat > 1).squeeze()
            sigma_num_hat[out_of_bound] = sigma_num_cur[out_of_bound]
            t_hat[out_of_bound] = t[out_of_bound]
            sigma_cat_hat = self.cat_schedule.total_noise(t_hat)
        else:
            t_hat = t
            sigma_num_hat = sigma_num_cur
            sigma_cat_hat = sigma_cat_cur
                
        # Sample priors for the continuous dimensions
        z_norm = torch.randn((b, self.num_numerical_features), device=device) * sigma_num_cur[-1] 
            
        # Sample priors for the discrete dimensions
        has_cat = len(self.num_classes) > 0
        z_cat = torch.zeros((b, 0), device=device).float()      # the default values for categorical sample if the dataset has no categorical entry
        if has_cat:
            z_cat = self._sample_masked_prior(
                b,
                len(self.num_classes),
            )
        
        pbar = tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps)
        pbar.set_description(f"Sampling Progress")
        for i in pbar:                  
            z_norm, z_cat, q_xs = self.edm_update(
                z_norm, z_cat, i, 
                t[i], t[i-1] if i > 0 else None, t_hat[i],
                sigma_num_cur[i], sigma_num_next[i], sigma_num_hat[i], 
                sigma_cat_cur[i], sigma_cat_next[i], sigma_cat_hat[i],
            )
        
        if not torch.all(z_cat < self.mask_index):      # catch any update result in the mask class or the dummy classes
            error_index = torch.any(z_cat >= self.mask_index, dim=-1).nonzero()
            error_z_cat = z_cat[error_index]
            error_q_xs = q_xs[error_index]
            print(error_index)
            print(error_z_cat)
            print(error_q_xs)
        assert torch.all(z_cat < self.mask_index)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample
    
    def sample_all(self, num_samples, batch_size, keep_nan_samples=False):        
        b = batch_size

        all_samples = []
        num_generated = 0
        while num_generated < num_samples:
            print(f"Samples left to generate: {num_samples-num_generated}")
            sample = self.sample(b)
            mask_nan = torch.any(sample.isnan(), dim=1)
            if keep_nan_samples:
                # If the sample instances that contains Nan are decided to be kept, the row with Nan will be foreced to all zeros
                sample = sample * (~mask_nan)[:, None]
            else:
                # Otherwise the instances with Nan will be eliminated
                sample = sample[~mask_nan]

            all_samples.append(sample)
            num_generated += sample.shape[0]

        x_gen = torch.cat(all_samples, dim=0)[:num_samples]

        return x_gen
    
    def q_xt(self, x, move_chance, strategy='hard'):
        """Computes the noisy sample xt.

        Args:
        x: int torch.Tensor with shape (batch_size,
            diffusion_model_input_length), input. 
        move_chance: float torch.Tensor with shape (batch_size, 1).
        """
        move_indices = torch.rand(*x.shape, device=x.device) < move_chance
        xt = torch.where(move_indices, self.mask_index, x)
        xt_soft = self.to_one_hot(xt).to(move_chance.dtype)
        return xt, xt_soft
    
    def q_xt_pairwise(self, x_i, x_j, move_chance, strategy='hard'):
        """Computes the noisy samples for a pair (x_i, x_j).
        
        Both x_i and x_j will receive the same masking decision (either both masked or both not masked).
        
        Args:
            x_i: int torch.Tensor with shape (batch_size, num_cat_features), higher-ranked doc labels
            x_j: int torch.Tensor with shape (batch_size, num_cat_features), lower-ranked doc labels
            move_chance: float torch.Tensor with shape (batch_size, 1).
            
        Returns:
            x_t: mask indicator tensor (batch_size, 1), same for both pairs
            x_i_t_soft: (batch_size, 2) where [:, 0] = x_i label, [:, 1] = x_t mask indicator
            x_j_t_soft: (batch_size, 2) where [:, 0] = x_j label, [:, 1] = x_t mask indicator
        """
        move_indices = torch.rand(*x_i.shape, device=x_i.device) < move_chance
        
        x_t = torch.where(move_indices, 1, 0).float()    # mask indicator
        
        x_i_t = torch.where(move_indices, self.mask_index, x_i)
        x_j_t = torch.where(move_indices, self.mask_index, x_j)
        
        x_i_t_soft = self.to_one_hot(x_i_t).to(move_chance.dtype)
        x_j_t_soft = self.to_one_hot(x_j_t).to(move_chance.dtype)
        
        return x_t, x_i_t_soft, x_j_t_soft
    
    
    def _subs_parameterization(self, unormalized_prob, xt):
        # Add a dummy class to the unnormalized probabilities to account for the mask category.
        unormalized_prob = torch.cat((
            unormalized_prob,
            torch.zeros(unormalized_prob.shape[0], 1, device=unormalized_prob.device, dtype=unormalized_prob.dtype)
        ), dim=-1)
        
        # log prob at the mask index = - infinity
        unormalized_prob = self.pad(unormalized_prob, self.neg_infinity)
        
        unormalized_prob[:, range(unormalized_prob.shape[1]), self.mask_index] += self.neg_infinity
        
        # Take log softmax on the unnormalized probabilities to the logits
        logits = unormalized_prob - torch.logsumexp(unormalized_prob, dim=-1, keepdim=True)
        
        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = (xt != self.mask_index)    # (bs, K)
        logits[unmasked_indices] = self.neg_infinity 
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits
    
    def pad(self, x, pad_value):
        """
        Converts a concatenated tensor of class probabilities into a padded matrix, 
        where each sub-tensor is padded along the last dimension to match the largest 
        category size (max number of classes).

        Args:
            x (Tensor): The input tensor containing concatenated probabilities for all the categories in x_cat. 
                        [bs, sum(num_classes_w_mask)]
            pad_value (float): The value filled into the dummy entries, which are padded to ensure all sub-tensors have equal size 
                            along the last dimension.

        Returns:
            Tensor: A new tensorwith
                    [bs, len(num_classes_w_mask), max(num_classes_w_mask)), num_categories]
        """
        splited = torch.split(x, self.num_classes_w_mask, dim=-1)
        max_K = max(self.num_classes_w_mask)
        padded_ = [
            torch.cat((
                t, 
                pad_value*torch.ones(*(t.shape[:-1]), max_K-t.shape[-1], dtype=t.dtype, device=t.device)
            ), dim=-1) 
        for t in splited]
        out = torch.stack(padded_, dim=-2)
        return out
    
    def to_one_hot(self, x_cat):
        x_cat_oh = torch.cat(
            [F.one_hot(x_cat[:, i], num_classes=self.num_classes[i]+1,) for i in range(len(self.num_classes))], 
            dim=-1
        )
        return x_cat_oh
    
    def _absorbed_closs(self, model_output, x0, sigma):
        """
            alpha: (bs,)
        """
        log_p_theta = -torch.gather(model_output, -1, x0[:, :, None]).squeeze(-1)

        alpha = torch.exp(-sigma)
        elbo_weight = 1/(1-alpha)
        
        loss = elbo_weight * log_p_theta
        
        return loss
    
    def _ranknet_loss(self, score_i, score_j, mask, sigma):
        """
        RankNet loss: -log(sigmoid(score_i - score_j))
        where score_i should be higher than score_j.
        
        Args:
            score_i: predicted scores for higher-ranked docs
            score_j: predicted scores for lower-ranked docs
            mask: optional mask tensor (1 = compute loss, 0 = ignore)
                  Only masked samples contribute to the loss.
        """
        diff = score_i - score_j
        loss = -torch.log(torch.sigmoid(diff) + 1e-10)
        
        alpha = torch.exp(-sigma)
        elbo_weight = 1/(1-alpha)
        
        loss = elbo_weight * loss
        loss = loss * mask
            
        return loss
    
    def _sample_masked_prior(self, *batch_dims):
        return self.mask_index[None,:] * torch.ones(*batch_dims, dtype=torch.int64, device=self.mask_index.device)
        
    def _mdlm_update(self, log_p_x0, x, alpha_t, alpha_s):
        """
            # t: (bs,)
            log_p_x0: (bs, K, K_max)
            # alpha_t: (bs,)
            # alpha_s: (bs,)
            alpha_t: (bs, 1/K_cat)
            alpha_s: (bs,1/K_cat)
        """
        move_chance_t = 1 - alpha_t
        move_chance_s = 1 - alpha_s     
        move_chance_t = move_chance_t.unsqueeze(-1)
        move_chance_s = move_chance_s.unsqueeze(-1)
        assert move_chance_t.ndim == log_p_x0.ndim
        # Technically, this isn't q_xs since there's a division
        # term that is missing. This division term doesn't affect
        # the samples.
        # There is a noremalizing term is (1-\alpha_t) who's responsility is to ensure q_xs is normalized. 
        # However, omiting it won't make a difference for the Gumbel-max sampling trick in  _sample_categorical()
        q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)
        q_xs[:, range(q_xs.shape[1]), self.mask_index] = move_chance_s[:, :, 0]
        
        # Important: make sure that prob of dummy classes are exactly 0
        dummy_mask = torch.tensor([[(1 if i <= mask_idx else 0) for i in range(max(self.mask_index+1))] for mask_idx in self.mask_index], device=q_xs.device)
        dummy_mask = torch.ones_like(q_xs) * dummy_mask
        q_xs *= dummy_mask
        
        _x = self._sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)
        
        z_cat = copy_flag * x + (1 - copy_flag) * _x
        if not torch.all(z_cat <= self.mask_index):     # catch any update result in the dummy classes
            error_index = torch.any(z_cat > self.mask_index, dim=-1).nonzero()
            error_z_cat = z_cat[error_index]
            error_q_xs = q_xs[error_index]
            print(error_index)
            print(error_z_cat)
            print(error_q_xs)
        return copy_flag * x + (1 - copy_flag) * _x, q_xs

    def _sample_categorical(self, categorical_probs):
        gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log())
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    
    def sample_ctime_noise(self, batch):
        if self.noise_dist == 'log_norm':
            rnd_normal = torch.randn(batch.shape[0], device=batch.device)
            sigma = (rnd_normal * self.noise_dist_params['P_std'] + self.noise_dist_params['P_mean']).exp()
        else:
            raise NotImplementedError(f"The noise distribution--{self.noise_dist}-- is not implemented for CTIME ")
        return sigma

    def _edm_loss(self, D_yn, y, sigma):
        weight = (sigma ** 2 + self.edm_params['sigma_data'] ** 2) / (sigma * self.edm_params['sigma_data']) ** 2
    
        target = y
        loss = weight * ((D_yn - target) ** 2)

        return loss
    
    def edm_update(
            self, x_num_cur, x_cat_cur, i, 
            t_cur, t_next, t_hat,
            sigma_num_cur, sigma_num_next, sigma_num_hat, 
            sigma_cat_cur, sigma_cat_next, sigma_cat_hat, 
        ):
        """
        i = T-1,...,0
        """
        
        b = x_num_cur.shape[0]
        has_cat = len(self.num_classes) > 0
        
        # Get x_num_hat by move towards the noise by a small step
        x_num_hat = x_num_cur + (sigma_num_hat ** 2 - sigma_num_cur ** 2).sqrt() * S_noise * torch.randn_like(x_num_cur)
        # Get x_cat_hat
        move_chance = -torch.expm1(sigma_cat_cur - sigma_cat_hat)    # the incremental move change is 1 - alpha_t/alpha_s = 1 - exp(sigma_s - sigma_t)
        x_cat_hat, _ = self.q_xt(x_cat_cur, move_chance) if has_cat else (x_cat_cur, x_cat_cur)

        # Get predictions
        x_cat_hat_oh = self.to_one_hot(x_cat_hat).to(x_num_hat.dtype) if has_cat else x_cat_hat
        denoised, raw_logits = self._denoise_fn(
            x_num_hat.float(), x_cat_hat_oh,
            t_hat.squeeze().repeat(b), sigma=sigma_num_hat.unsqueeze(0).repeat(b,1)  # sigma accepts (bs, K_num)
        )
        
        # Euler step
        d_cur = (x_num_hat - denoised) / sigma_num_hat
        x_num_next = x_num_hat + (sigma_num_next - sigma_num_hat) * d_cur
        
        # Unmasking
        x_cat_next = x_cat_cur
        q_xs = torch.zeros_like(x_cat_cur).float()
        if has_cat:
            logits = self._subs_parameterization(raw_logits, x_cat_hat)
            alpha_t = torch.exp(-sigma_cat_hat).unsqueeze(0).repeat(b,1)
            alpha_s = torch.exp(-sigma_cat_next).unsqueeze(0).repeat(b,1)
            x_cat_next, q_xs = self._mdlm_update(logits, x_cat_hat, alpha_t, alpha_s)
            if i == 0:
                x_cat_next = logits[:, :, 1]
        
        # Apply 2nd order correction.
        if self.sampler_params['second_order_correction']:
            if i > 0:
                x_cat_hat_oh = self.to_one_hot(x_cat_hat).to(x_num_next.dtype) if has_cat else x_cat_hat
                denoised, raw_logits = self._denoise_fn(
                    x_num_next.float(), x_cat_hat_oh,
                    t_next.squeeze().repeat(b), sigma=sigma_num_next.unsqueeze(0).repeat(b,1)
                )
                d_prime = (x_num_next - denoised) / sigma_num_next
                x_num_next = x_num_hat + (sigma_num_next - sigma_num_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
        return x_num_next, x_cat_next, q_xs


    def sample_impute(self, x_num, x_cat, num_mask_idx, cat_mask_idx, impute_condition):
        self.num_mask_idx = num_mask_idx
        self.cat_mask_idx = cat_mask_idx
        
        b = x_num.size(0)
        device = self.device
        dtype = torch.float32

        # Create masks, true for the missing columns
        num_mask = [i in num_mask_idx for i in range(self.num_numerical_features)]
        cat_mask = [i in cat_mask_idx for i in range(len(self.num_classes))]
        num_mask = torch.tensor(num_mask).to(x_num.device).to(x_num.dtype)
        cat_mask = torch.tensor(cat_mask).to(x_cat.device).to(x_cat.dtype)

        # Create the chain of t
        t = torch.linspace(0,1,self.num_timesteps, dtype=dtype, device=device)      # times = 0.0,...,1.0
        t = t[:, None]
        
        # Compute the chains of sigma
        sigma_num_cur = self.num_schedule.total_noise(t)
        sigma_cat_cur = self.cat_schedule.total_noise(t)
        sigma_num_next = torch.zeros_like(sigma_num_cur)
        sigma_num_next[1:] = sigma_num_cur[0:-1]
        sigma_cat_next = torch.zeros_like(sigma_cat_cur)
        sigma_cat_next[1:] = sigma_cat_cur[0:-1]
        
        t_hat = t
        sigma_num_hat = sigma_num_cur
        sigma_cat_hat = sigma_cat_cur

        # Sample priors for the continuous dimensions
        if impute_condition == "x_t":
            z_norm = x_num + torch.randn((b, self.num_numerical_features), device=device) * sigma_num_cur[-1]   # z_{t_max} = x_0(masked) + sigma_max*epsilon
        elif impute_condition == "x_0":
            z_norm = x_num
            
        # Sample priors for the discrete dimensions
        has_cat = len(self.num_classes) > 0
        z_cat = torch.zeros((b, 0), device=device).float()      # the default values for categorical sample if the dataset has no categorical entry
        if has_cat:
            if impute_condition == "x_t":
                z_cat = self._sample_masked_prior(b, len(self.num_classes))   # z_{t_max} is still all pushed to [MASK]
            elif impute_condition == "x_0":
                z_cat = x_cat
        
        for i in tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps, desc="Sampling Progress"):
            # Get known parts by Forward Flow
            if impute_condition == "x_t":
                z_norm_known = x_num + torch.randn((b, self.num_numerical_features), device=device) * sigma_num_next[i]
                move_chance = 1 - torch.exp(-sigma_cat_next[i]) if i < (self.num_timesteps-1) else torch.ones_like(sigma_cat_next[i])     # force move_chance to be 1 for the first iteration
                z_cat_known, _ = self.q_xt(x_cat, move_chance)
            elif impute_condition == "x_0":
                z_norm_known = x_num
                z_cat_known = x_cat
            
            # Get unknown by Reverse Step
            z_norm_unknown, z_cat_unknown, q_xs = self.edm_update(
                z_norm, z_cat, i, 
                t[i], t[i-1] if i > 0 else None, t_hat[i],
                sigma_num_cur[i], sigma_num_next[i], sigma_num_hat[i], 
                sigma_cat_cur[i], sigma_cat_next[i], sigma_cat_hat[i],
            )
            z_norm = (1 - num_mask)  * z_norm_known + num_mask * z_norm_unknown
            z_cat = (1 - cat_mask) * z_cat_known + cat_mask * z_cat_unknown
            z_cat = z_cat_unknown
        
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample
    