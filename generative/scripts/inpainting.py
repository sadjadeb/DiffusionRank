import torch
import os
import numpy as np
import zero
import lib
from tab_ddpm import GaussianMultinomialDiffusion, ohe_to_categories
from utils_train import make_dataset
from tab_ddpm.modules import MLPDiffusion
import argparse
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils_train import plot_inpainting_outputs
from utils import calculate_metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def index_to_log_onehot(x, num_classes):
    onehots = []
    for i in range(len(num_classes)):
        onehots.append(F.one_hot(x[:, i], num_classes[i]))
 
    x_onehot = torch.cat(onehots, dim=1)
    log_onehot = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_onehot


class InPainter:
    def __init__(self, diffusion, strategy, dataset, parent_dir, test_loader, test_loader_idx, device):
        self.diffusion = diffusion
        self.strategy = strategy
        self.dataset = dataset
        self.parent_dir = parent_dir
        self.device = device
        self.test_loader = test_loader
        self.test_loader_idx = test_loader_idx

    @torch.no_grad()
    def run_loop(self):
        device = self.diffusion.log_alpha.device
        has_cat = self.diffusion.num_classes[0] != 0
        
        X_idx = []
        X_predicted = []
        X_num_noisy = []
        for batch, batch_idx in tqdm(zip(self.test_loader, self.test_loader_idx), desc='Batches', total=len(self.test_loader), position=0):
            if type(batch) == tuple:
                batch = batch[0]
            
            b = batch.size(0)
            batch = batch.to(device)
            
            x_num = batch[:, :self.diffusion.num_numerical_features]
            x_cat = batch[:, self.diffusion.num_numerical_features:]
            
            t_max = torch.full((b,), self.diffusion.num_timesteps - 1, device=device, dtype=torch.long)
            
            # it automatically adds noise to the given input for t number of steps
            x_num_t = self.diffusion.gaussian_q_sample(x_num, t_max)
            x_num_noisy = x_num_t.clone()
            
            log_x_cat_t = x_cat
            if has_cat:
                log_x_cat = index_to_log_onehot(x_cat.long(), self.diffusion.num_classes)
                log_x_cat_t = self.diffusion.q_sample(log_x_start=log_x_cat, t=t_max)
                
            if has_cat:
                original_features = batch[:, :self.diffusion.num_numerical_features]
            else:
                original_features = batch[:, 1:self.diffusion.num_numerical_features]
            
            if self.strategy == 'full_noise':
                for i in range(self.diffusion.num_timesteps - 1, -1, -1):
                    t = torch.full((b,), i, device=device, dtype=torch.long)
                    model_out = self.diffusion._denoise_fn(
                        torch.cat([x_num_t, log_x_cat_t], dim=1).float(),
                        t,
                    )
                    
                    model_out_num = model_out[:, :self.diffusion.num_numerical_features]
                    model_out_cat = model_out[:, self.diffusion.num_numerical_features:]
                    x_num_t = self.diffusion.gaussian_p_sample(model_out_num, x_num_t, t, clip_denoised=False)['sample']
                    
                    if has_cat:
                        log_x_cat_t = self.diffusion.p_sample(model_out_cat, log_x_cat_t, t, {})
                
            elif self.strategy == 'renoise_features':
                fixed_noise = torch.randn_like(original_features)
                
                for i in range(self.diffusion.num_timesteps - 1, -1, -1):
                    t = torch.full((b,), i, device=device, dtype=torch.long)
                    model_out = self.diffusion._denoise_fn(
                        torch.cat([x_num_t, log_x_cat_t], dim=1).float(),
                        t,
                    )
                    
                    model_out_num = model_out[:, :self.diffusion.num_numerical_features]
                    model_out_cat = model_out[:, self.diffusion.num_numerical_features:]
                    x_num_t = self.diffusion.gaussian_p_sample(model_out_num, x_num_t, t, clip_denoised=False)['sample']
                    
                    if has_cat:
                        log_x_cat_t = self.diffusion.p_sample(model_out_cat, log_x_cat_t, t, {})
                    
                    noisy_features_t = self.diffusion.gaussian_q_sample(original_features, t, noise=fixed_noise)
                    if has_cat:
                        x_num_t[:, :self.diffusion.num_numerical_features] = noisy_features_t
                    else:
                        x_num_t[:, 1:self.diffusion.num_numerical_features] = noisy_features_t
                    
                
            elif self.strategy == 'original_features':
                # revert the original features
                if has_cat:
                    x_num_t[:, :self.diffusion.num_numerical_features] = original_features
                else:
                    x_num_t[:, 1:self.diffusion.num_numerical_features] = original_features
                
                for i in range(self.diffusion.num_timesteps - 1, -1, -1):
                    t = torch.full((b,), i, device=device, dtype=torch.long)
                    model_out = self.diffusion._denoise_fn(
                        torch.cat([x_num_t, log_x_cat_t], dim=1).float(),
                        t,
                    )
                    
                    model_out_num = model_out[:, :self.diffusion.num_numerical_features]
                    model_out_cat = model_out[:, self.diffusion.num_numerical_features:]
                    x_num_t = self.diffusion.gaussian_p_sample(model_out_num, x_num_t, t, clip_denoised=False)['sample']
                    
                    if has_cat:
                        log_x_cat_t = self.diffusion.p_sample(model_out_cat, log_x_cat_t, t, {})
                    
                    if has_cat:
                        x_num_t[:, :self.diffusion.num_numerical_features] = original_features
                    else:
                        x_num_t[:, 1:self.diffusion.num_numerical_features] = original_features
                        
            x_cat_t = log_x_cat_t
            if has_cat:
                x_ohe = torch.exp(log_x_cat_t).round()
                x_cat_t = ohe_to_categories(x_ohe, self.diffusion.num_classes)
            
            X_idx.append(batch_idx)
            X_predicted.append(torch.cat([x_num_t, x_cat_t.float()], dim=1))
            X_num_noisy.append(x_num_noisy)
            
            
        X_idx = torch.cat(X_idx, dim=0).cpu()
        X_predicted = torch.cat(X_predicted, dim=0).cpu()
        X_num_noisy = torch.cat(X_num_noisy, dim=0).cpu()
        
        return X_idx, X_predicted, X_num_noisy
        
    
    def inverse_transform_predicted(self, X_predicted):
        X_cat = X_predicted[:, self.diffusion.num_numerical_features:]
        
        if self.diffusion.num_numerical_features < X_predicted.shape[1]:
            # if T_dict['cat_encoding'] == 'one-hot':
            #     X_predicted[:, self.diffusion.num_numerical_features:] = to_good_ohe(D.cat_transform.steps[0][1], X_predicted[:, self.diffusion.num_numerical_features:])
            
            # This function maps the one-hot encoded values to the original values
            # in our case, 0 becomes '0', 1 becomes '1', etc.
            # X_cat = self.dataset.cat_transform.inverse_transform(X_predicted[:, self.diffusion.num_numerical_features:])
            pass

        if self.dataset.num_transform is not None:
            X_num = self.dataset.num_transform.inverse_transform(X_predicted[:, :self.diffusion.num_numerical_features])
        else:
            X_num = X_predicted[:, :self.diffusion.num_numerical_features]
            
        X_inversed = np.concatenate([X_num, X_cat], axis=1)
            
        return X_inversed
    
    def write_outputs(self, X_idx, X_predicted, y_pred):
        np.save(os.path.join(self.parent_dir, 'X_num_inpainted'), X_predicted[:, :self.diffusion.num_numerical_features])
        np.save(os.path.join(self.parent_dir, 'X_cat_inpainted'), X_predicted[:, self.diffusion.num_numerical_features:])
        np.save(os.path.join(self.parent_dir, 'X_idx'), X_idx)
        np.save(os.path.join(self.parent_dir, 'y_pred'), y_pred)

       
    def evaluate_results(self, X_idx, y_pred, y_true):
        results = {}
        for idx, label_t, label_p in zip(X_idx, y_true, y_pred):
            idx = idx.item()
            if idx not in results:
                results[idx] = []
            results[idx].append((label_t, label_p))
        
        avgndcg, avgp = calculate_metrics(results)
        
        has_cat = self.diffusion.num_classes[0] != 0
        if has_cat:
            unique, counts = np.unique(y_pred, return_counts=True)
            print("Predicted labels and their counts:")
            for value, count in zip(unique, counts):
                print(f"Value: {value}, Count: {count}")
            print()
            
            # Accuracy
            accuracy = accuracy_score(y_true, y_pred)
            print(f"Accuracy: {accuracy:.4f}")

            # Precision (Macro and Micro Average)
            precision_macro = precision_score(y_true, y_pred, average='macro')
            precision_micro = precision_score(y_true, y_pred, average='micro')
            print(f"Precision (Macro): {precision_macro:.4f} - Precision (Micro): {precision_micro:.4f}")

            # Recall (Macro and Micro Average)
            recall_macro = recall_score(y_true, y_pred, average='macro')
            recall_micro = recall_score(y_true, y_pred, average='micro')
            print(f"Recall (Macro): {recall_macro:.4f} - Recall (Micro): {recall_micro:.4f}")

            # F1 Score (Macro and Micro Average)
            f1_macro = f1_score(y_true, y_pred, average='macro')
            f1_micro = f1_score(y_true, y_pred, average='micro')
            print(f"F1 Score (Macro): {f1_macro:.4f} - F1 Score (Micro): {f1_micro:.4f}")

            # Confusion Matrix
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_true, y_pred))
            print()
        else:
            # calculate the mse between true and predicted labels
            mse = np.mean((y_true - y_pred) ** 2)
            y_true_min, y_true_max = np.min(y_true), np.max(y_true)
            y_pred_min, y_pred_max = np.min(y_pred), np.max(y_pred)
            print(f'MSE: {mse}')
            print(f'y_true_max: {y_true_max:.2f}, y_true_min: {y_true_min:.2f}')
            print(f'y_pred_max: {y_pred_max:.2f}, y_pred_min: {y_pred_min:.2f}')
        
        with open(os.path.join(self.parent_dir, f'results.{self.strategy}.txt'), 'w') as f:
            f.write('qid\ttrue\tpred\n')
            for idx, labels in results.items():
                for label_t, label_p in labels:
                    f.write(f'{idx}\t{label_t}\t{label_p:.6f}\n')

        return avgndcg, avgp



def inpaint(
    parent_dir,
    real_data_path,
    batch_size = 4096,
    model_params = None,
    model_path = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    num_numerical_features = 0,
    device = "cpu",
    strategy = 'full_noise',
    y_index = 0
):
    
    real_data_path = os.path.normpath(real_data_path)
    parent_dir = os.path.normpath(parent_dir)

    zero.improve_reproducibility(42)

    T = lib.Transformations(**T_dict)

    D = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
    )

    K = np.array(D.get_category_sizes('test'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])
    
    num_numerical_features_ = D.X_num['test'].shape[1] if D.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features_
    model_params['d_in'] = d_in
    
    model = MLPDiffusion(**model_params)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    has_cat = model.num_classes != 0
    
    if has_cat:
        y_index = -1
    
    split = 'test'
    X = D.X_num[split]
    
    # Get the original labels to evaluate inpainting; As the features are normalized, we need to inverse transform them
    if D.num_transform is not None:
        X_num_unnorm = D.num_transform.inverse_transform(X[:, :num_numerical_features_])
    else:
        X_num_unnorm = X
        
    if D.X_cat is not None:
        X = torch.from_numpy(np.concatenate([D.X_num[split], D.X_cat[split]], axis=1)).float()
    else:
        X = torch.from_numpy(D.X_num[split]).float()
    y = torch.from_numpy(D.y[split])
        
    # replace real labels with random labels
    current_labels = X[:, y_index]
    labels_unique = torch.unique(current_labels)
    random_labels = labels_unique[torch.randint(0, len(labels_unique), (X.shape[0],))]
    X[:, y_index] = random_labels
    
    test_loader = lib.FastTensorDataLoader(X, y, batch_size=batch_size)
    
    X_idx = torch.from_numpy(np.load(os.path.join(real_data_path, f"idx_{split}.npy"), allow_pickle=True))
    test_loader_idx = DataLoader(X_idx, batch_size=batch_size, shuffle=False)

    diffusion = GaussianMultinomialDiffusion(
        K,
        num_numerical_features=num_numerical_features_,
        denoise_fn=model,
        num_timesteps=num_timesteps, 
        gaussian_loss_type=gaussian_loss_type,
        scheduler=scheduler,
        device=device
    )

    diffusion.to(device)
    diffusion.eval()
    
    inpainter = InPainter(diffusion, strategy, D, parent_dir, test_loader, test_loader_idx, device)
    X_idx, X_predicted, X_predicted_noisy = inpainter.run_loop()
    
    X_predicted_inversed = inpainter.inverse_transform_predicted(X_predicted)
    
    if has_cat:
        y_true = D.y['test']
    else:
        y_true = X_num_unnorm[:, y_index]
        
    y_pred = X_predicted_inversed[:, y_index]
    X_num_pred_inv = X_predicted_inversed[:, :num_numerical_features_]
    for i in range(num_numerical_features_):
        mse = np.mean((X_num_unnorm[:, i] - X_num_pred_inv[:, i]) ** 2)
        print(f'MSE for index {i}: {mse:.6f}')
        
    avgndcg, avgp = inpainter.evaluate_results(X_idx, y_pred, y_true)
    print(f'strategy: {strategy}, avgndcg: {avgndcg}, avgp: {avgp}')
    avgndcg, avgp = inpainter.evaluate_results(X_idx, random_labels.numpy(), y_true)
    print(f'strategy: random_labels, avgndcg: {avgndcg}, avgp: {avgp}')
    
    plot_inpainting_outputs(parent_dir, y_index, D, X_num_unnorm, X_num_pred_inv, X_predicted_noisy, y_pred, strategy)
    inpainter.write_outputs(X_idx, X_predicted, y_pred)
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_model_path', metavar='FILE')
    parser.add_argument('--strategy', type=str, default='full_noise', choices=['full_noise', 'renoise_features', 'original_features'])
    parser.add_argument('--y_index', type=int, default=0)
    args = parser.parse_args()
    
    print(f'Inpainting with strategy: {args.strategy} by {args.trained_model_path} for index {args.y_index}')

    config_path = os.path.join(args.trained_model_path, 'config.toml')
    model_path = os.path.join(args.trained_model_path, 'model.best.pt')
    raw_config = lib.load_config(config_path)
    
    device = torch.device(raw_config['device'])
    raw_config['parent_dir'] = raw_config['parent_dir']
    dataset = raw_config['real_data_path'].split('/')[-2]
    
    inpaint(
        **raw_config['diffusion_params'],
        parent_dir=raw_config['parent_dir'],
        real_data_path=raw_config['real_data_path'],
        model_path=model_path,
        model_params=raw_config['model_params'],
        T_dict=raw_config['train']['T'],
        num_numerical_features=raw_config['num_numerical_features'],
        device=device,
        strategy=args.strategy,
        y_index=args.y_index
    )
