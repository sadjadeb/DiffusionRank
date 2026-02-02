import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from utils import set_all_seeds, calculate_metrics
from model import DNN
from sklearn.preprocessing import QuantileTransformer
from copy import deepcopy
import argparse
import random

# Set random seeds for reproducibility
seed = 42
set_all_seeds(seed)

# get k from command line arguments
parser = argparse.ArgumentParser(description='Train and test a pairwise neural network with noisy data for Learning-to-Rank tasks')
parser.add_argument('--dataset', type=str, required=True, choices=['MQ2007', 'MQ2008', 'MSLR-WEB10K', 'MSLR-WEB30K'], help='Dataset to use for the experiment')
parser.add_argument('--task', type=str, choices=['train', 'test'], help='Task to run: train or test') 
parser.add_argument('--k', type=float, default=1.0, help='Fraction k for the dataset')
parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging')
parser.add_argument('--exp_name', type=str, default=None, help='Experiment name for logging')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to the model checkpoint to load')
parser.add_argument('--num_hidden_nodes', type=int, required=True, help='Number of hidden nodes in the model')
parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate for the optimizer')
args = parser.parse_args()

dataset = args.dataset
k = args.k
print(f'Running experiment for dataset: {dataset}, k: {k}')

if args.task == 'test' and args.checkpoint is None:
    raise ValueError("Checkpoint must be provided for testing.")
if args.task == 'test':
    args.no_wandb = True

# Set hyperparameters
device = torch.device("cuda:0")
features_count = 136 if 'MSLR' in dataset else 46
data_normalization = 'quantile'
num_epochs = 2000 if 'MSLR' in dataset else 5000
dropout_rate = 0.1
learning_rate = args.lr
num_hidden_nodes = args.num_hidden_nodes
batch_size = 4096

features_count += 1  # Add 1 for time feature

wandb.init(project=f"DiffusionRank_{dataset}_final", 
           name=f"disc_pairwise_perturbed_k{k}" + (f"_{args.exp_name}" if args.exp_name else ""), 
           mode='disabled' if args.no_wandb else 'online')
wandb.config.update({
    'features_count': features_count,
    'data_normalization': data_normalization,
    'num_epochs': num_epochs,
    'dropout_rate': dropout_rate,
    'learning_rate': learning_rate,
    'num_hidden_nodes': num_hidden_nodes,
    'batch_size': batch_size,
    'k': k,
})

# Set data paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_root, 'data', dataset, 'by_fraction', 'Fold1', f'k{k}')


X_train = np.load(os.path.join(data_dir, 'X_num_train.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
idx_train = np.load(os.path.join(data_dir, 'idx_train.npy'))

X_val = np.load(os.path.join(data_dir, 'X_num_val.npy'))
y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
idx_val = np.load(os.path.join(data_dir, 'idx_val.npy'))

X_test = np.load(os.path.join(data_dir, 'X_num_test.npy'))
y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
idx_test = np.load(os.path.join(data_dir, 'idx_test.npy'))

# Normalize the data
if data_normalization == 'quantile':
    normalizer = QuantileTransformer(
                output_distribution='normal',
                n_quantiles=max(min(X_train.shape[0] // 30, 1000), 10),
                subsample=int(1e9),
                random_state=seed,
            )
    X_train = normalizer.fit_transform(X_train)
    X_val = normalizer.transform(X_val)
    X_test = normalizer.transform(X_test)

# For pairwise training, we need to organize data by query ID
# Keep training data as numpy for perturbation, convert to tensor later
X_train_tensor = torch.from_numpy(X_train).float().to(device)

# Organize training data by query ID for efficient pair sampling
train_data_by_qid = {}
for i in range(len(X_train)):
    qid = idx_train[i]
    if qid not in train_data_by_qid:
        train_data_by_qid[qid] = {'indices': []}
    train_data_by_qid[qid]['indices'].append(i)

# Also store labels separately for pair generation
train_labels = y_train
train_indices = idx_train

# Filter out queries with only one document (can't form pairs)
num_queries_before = len(train_data_by_qid)
train_data_by_qid = {qid: data for qid, data in train_data_by_qid.items() 
                     if len(data['indices']) > 1}
num_queries_after = len(train_data_by_qid)
num_filtered = num_queries_before - num_queries_after
print(f"Organized training data into {num_queries_after} queries with multiple documents (filtered out {num_filtered} queries with only one document)")

# add zeros to the val and test sets for the time feature
X_val = np.concatenate((X_val, np.zeros((X_val.shape[0], 1))), axis=1)
X_test = np.concatenate((X_test, np.zeros((X_test.shape[0], 1))), axis=1)

# Create a dataloader for the validation data using pytorch
val_reader = torch.utils.data.TensorDataset(torch.from_numpy(X_val).float().to(device), torch.from_numpy(y_val).long(), torch.from_numpy(idx_val).long())
val_reader_iter = torch.utils.data.DataLoader(val_reader, batch_size=batch_size, shuffle=False)
# Create a dataloader for the test data using pytorch
test_reader = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float().to(device), torch.from_numpy(y_test).long(), torch.from_numpy(idx_test).long())
test_reader_iter = torch.utils.data.DataLoader(test_reader, batch_size=batch_size, shuffle=False)

# Create model, optimizer, and loss function
net = DNN(input_dim=features_count, approach='pairwise', num_hidden_nodes=num_hidden_nodes, dropout_rate=dropout_rate).to(device)
optimizer = optim.AdamW(net.parameters(), lr=learning_rate)

# RankNet loss function
class RankNetLoss(nn.Module):
    """
    RankNet loss function for pairwise learning to rank.
    Loss = -log(sigmoid(s_i - s_j)) where document i should be ranked higher than document j.
    """
    def __init__(self):
        super(RankNetLoss, self).__init__()
    
    def forward(self, scores_i, scores_j):
        """
        Args:
            scores_i: scores for documents that should be ranked higher (shape: [batch_size])
            scores_j: scores for documents that should be ranked lower (shape: [batch_size])
        Returns:
            loss: RankNet loss
        """
        # Compute the difference
        diff = scores_i - scores_j
        # RankNet loss: -log(sigmoid(diff))
        loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()
        return loss

criterion = RankNetLoss()

# print number of parameters
num_params = sum(p.numel() for p in net.parameters())
print(f"Number of model parameters: {num_params}")

if args.checkpoint:
    checkpoint = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(checkpoint)
    print(f"Weights loaded from checkpoint: {args.checkpoint}")


def forward_diffusion_step(x0, t, beta_schedule):
    """
    Perturbs input x0 at timestep t using Gaussian noise, based on beta_schedule.

    Parameters:
    - x0: torch.Tensor, the original input tensor (e.g., shape (batch_size, n_features))
    - t: torch.Tensor or int, timestep index (0 <= t < T). If int, it will be converted to a tensor.
    - beta_schedule: np.ndarray, schedule of betas with shape (T,), where T is total timesteps

    Returns:
    - xt: torch.Tensor, the noisy version of x0 at timestep t
    - noise: torch.Tensor, the sampled Gaussian noise added
    """
    if isinstance(t, int):
        t = torch.tensor([t], dtype=torch.long)
    beta_t = beta_schedule[t]
    alpha_t = 1. - beta_t
    alpha_bar_t = torch.cumprod(alpha_t, dim=0)[t]  # Product of (1 - beta) up to t

    noise = torch.randn_like(x0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t).view(-1, 1)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t).view(-1, 1)

    xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
    return xt, noise


def generate_pairs(qid_data, X_perturbed, labels, batch_size, device):
    """
    Generate pairs of documents for pairwise training from the same query.
    For each pair, the first document should have a higher label than the second.
    Pairs with equal labels are skipped as they provide no learning signal.
    
    Args:
        qid_data: dictionary mapping qid to document indices
        X_perturbed: perturbed feature tensor (with time feature)
        labels: label array
        batch_size: number of pairs to generate
        device: torch device
    """
    pairs_i = []
    pairs_j = []
    
    # Sample pairs from the same query
    qids = list(qid_data.keys())
    attempts = 0
    max_attempts = batch_size * 10  # Prevent infinite loop
    
    while len(pairs_i) < batch_size and attempts < max_attempts:
        attempts += 1
        qid = random.choice(qids)
        query_indices = qid_data[qid]['indices']
        n_docs = len(query_indices)
        
        # Need at least 2 documents to form a pair
        if n_docs < 2:
            continue
        
        # Sample two different documents
        local_idx_i, local_idx_j = random.sample(range(n_docs), 2)
        idx_i = query_indices[local_idx_i]
        idx_j = query_indices[local_idx_j]
        
        label_i = labels[idx_i]
        label_j = labels[idx_j]
        
        # Skip pairs with equal labels (no preference to learn)
        if label_i == label_j:
            continue
        
        # Ensure document i has higher label than document j
        if label_i < label_j:
            idx_i, idx_j = idx_j, idx_i
        
        pairs_i.append(X_perturbed[idx_i].cpu().numpy())
        pairs_j.append(X_perturbed[idx_j].cpu().numpy())
    
    if len(pairs_i) < batch_size:
        print(f"Warning: Only generated {len(pairs_i)} pairs out of {batch_size} requested")
    
    return torch.from_numpy(np.array(pairs_i)).float().to(device), torch.from_numpy(np.array(pairs_j)).float().to(device)


def train(net, X_perturbed):
    net.train()
    train_loss = 0.0
    num_batches = 0
    
    num_batches_per_epoch = max(1, len(X_train) // batch_size)
    for _ in range(num_batches_per_epoch):
        features_i, features_j = generate_pairs(train_data_by_qid, X_perturbed, train_labels, batch_size, device)
        
        # Get scores for both documents
        scores_i = net(features_i).squeeze()
        scores_j = net(features_j).squeeze()
        
        # Compute RankNet loss
        loss = criterion(scores_i, scores_j)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        num_batches += 1
    
    return train_loss / num_batches


def test(net, data_iter):
    net.eval()
    results = {}
    
    with torch.no_grad():
        for features, labels, qids in data_iter:
            # Get scores for individual documents
            scores = net(features).squeeze().data.cpu()
            
            row_cnt = len(qids)
            for i in range(row_cnt):
                qid = qids[i].item()
                if qid not in results:
                    results[qid] = []
                results[qid].append((labels[i].item(), scores[i].item()))
        
        avg_ndcg, avg_map = calculate_metrics(results)
        
        return {
            'map': avg_map,
            'ndcg': avg_ndcg,
            'results': results
        }



if __name__ == '__main__':
    print(f'Dataset: {dataset}')
    print(f'Number of learnable parameters: {net.parameter_count()}')
    
    # Beta schedule (e.g., linear from 1e-5 to 1e-3 over 50 steps)
    T = 50
    beta_schedule = torch.linspace(1e-5, 1e-3, T).to(device)
    
    if args.task == 'train':
        print('Start training the model with noisy data...')
        
        best_ndcg = float('-inf')
        best_model_state = None
        for epoch in range(num_epochs+1):
            # Train (skip actual training on epoch 0 for baseline)
            if epoch > 0:
                # Sample timestep t per sample in batch
                t = torch.randint(0, T, (X_train_tensor.size(0),), device=device)
                norm_t = t.float() / (T - 1)
                
                # Perturb inputs
                x_noisy, _ = forward_diffusion_step(X_train_tensor, t, beta_schedule)
                
                # Concatenate with time feature
                x_noisy = torch.cat((x_noisy, norm_t.view(-1, 1)), dim=1)
                
                train_loss = train(net, x_noisy)
            else:
                # For epoch 0, create unperturbed data with zero time feature
                x_noisy = torch.cat((X_train_tensor, torch.zeros(X_train_tensor.size(0), 1).to(device)), dim=1)
                train_loss = None
            
            # Evaluate on validation and test sets
            val_metrics = test(net, val_reader_iter)
            test_metrics = test(net, test_reader_iter)
            print(f'epoch: {epoch}, train_loss: {train_loss}, val_map: {val_metrics["map"]}, val_ndcg: {val_metrics["ndcg"]}, test_map: {test_metrics["map"]}, test_ndcg: {test_metrics["ndcg"]}')
            wandb.log({'loss/train_d_loss': train_loss,
                       'ranking_metrics/val_ndcg': val_metrics["ndcg"],
                       'ranking_metrics/val_map': val_metrics["map"],
                       'ranking_metrics/test_ndcg': test_metrics["ndcg"],
                       'ranking_metrics/test_map': test_metrics["map"],})
            
            # Track Best Model
            if val_metrics["ndcg"] > best_ndcg:
                best_ndcg = val_metrics["ndcg"]
                best_model_state = deepcopy(net.state_dict())

        final_model_save_path = os.path.join(project_root, 'discriminative', 'checkpoints', f'ltr.{dataset}.pairwise_perturbed.k{k}.{args.exp_name}.final.pt')
        best_model_save_path = os.path.join(project_root, 'discriminative', 'checkpoints', f'ltr.{dataset}.pairwise_perturbed.k{k}.{args.exp_name}.best.pt')

        # Save model
        torch.save(net.state_dict(), final_model_save_path)
        print(f'Final model saved to {final_model_save_path}')

        # Load best model before saving
        net.load_state_dict(best_model_state)
        torch.save(net.state_dict(), best_model_save_path)
        print(f'Best model saved to {best_model_save_path}')
    
    print('Evaluating the best model on test set...')
    test_metrics = test(net, test_reader_iter)
    print(f'Best Model Performance: test_map: {test_metrics["map"]}, test_ndcg: {test_metrics["ndcg"]}')
        
    # Save results
    results_save_path = os.path.join(project_root, 'discriminative', 'predictions', f'ltr.{dataset}.pairwise_perturbed.k{k}.{args.exp_name}.best.txt')
    with open(results_save_path, 'w') as f:
        f.write('qid true_label pred_label\n')
        for qid, values in test_metrics["results"].items():
            for true_label, pred_label in values:
                f.write(f'{qid} {true_label} {pred_label}\n')
    print('Results saved to {}'.format(results_save_path))

    wandb.finish()
