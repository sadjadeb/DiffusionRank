import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from utils import set_all_seeds, calculate_metrics
from model import DNN
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import ndcg_score
from copy import deepcopy
import argparse
import random

# Set random seeds for reproducibility
seed = 42
set_all_seeds(seed)

# get k from command line arguments
parser = argparse.ArgumentParser(description='Train and test a listwise neural network for Learning-to-Rank tasks')
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

wandb.init(project=f"DiffusionRank_{dataset}", 
           name=f"disc_listwise_k{k}" + (f"_{args.exp_name}" if args.exp_name else ""), 
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

# Organize training data by query ID for efficient query-level processing
train_data_by_qid = {}
for i in range(len(X_train)):
    qid = idx_train[i]
    if qid not in train_data_by_qid:
        train_data_by_qid[qid] = {'features': [], 'labels': []}
    train_data_by_qid[qid]['features'].append(X_train[i])
    train_data_by_qid[qid]['labels'].append(y_train[i])

# Filter out queries with only one document (can't form pairs or compute NDCG meaningful swaps)
num_queries_before = len(train_data_by_qid)
train_data_by_qid = {qid: data for qid, data in train_data_by_qid.items() if len(data['labels']) > 1}
num_queries_after = len(train_data_by_qid)
num_filtered = num_queries_before - num_queries_after
print(f"Organized training data into {num_queries_after} queries with multiple documents (filtered out {num_filtered} queries with only one document)")

# Create a dataloader for the validation and test data
val_reader = torch.utils.data.TensorDataset(torch.from_numpy(X_val).float().to(device), torch.from_numpy(y_val).long(), torch.from_numpy(idx_val).long())
val_reader_iter = torch.utils.data.DataLoader(val_reader, batch_size=batch_size, shuffle=False)

test_reader = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float().to(device), torch.from_numpy(y_test).long(), torch.from_numpy(idx_test).long())
test_reader_iter = torch.utils.data.DataLoader(test_reader, batch_size=batch_size, shuffle=False)

# Create model and optimizer
net = DNN(input_dim=features_count, approach='listwise', num_hidden_nodes=num_hidden_nodes, dropout_rate=dropout_rate).to(device)
optimizer = optim.AdamW(net.parameters(), lr=learning_rate)

# LambdaRank loss function
class LambdaRankLoss(nn.Module):
    """
    LambdaRank loss function.
    Computes pairwise RankNet loss scaled by the absolute change in NDCG.
    """
    def __init__(self):
        super(LambdaRankLoss, self).__init__()
    
    def forward(self, scores, labels):
        # Ensure 1D tensors
        scores = scores.squeeze()
        labels = labels.squeeze()
        
        num_docs_of_query = scores.shape[0]
        if num_docs_of_query < 2:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

        # Pairwise differences
        scores_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
        labels_diff = labels.unsqueeze(1) - labels.unsqueeze(0)
        
        # Mask for valid pairs where doc i is more relevant than doc j
        S_ij = (labels_diff > 0).float()
        
        if S_ij.sum() == 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

        # Calculate ideal DCG (IDCG)
        ideal_sort_idx = torch.argsort(labels, descending=True)
        ideal_ranks = torch.zeros_like(ideal_sort_idx, dtype=torch.float32, device=scores.device)
        ideal_ranks[ideal_sort_idx] = torch.arange(1, num_docs_of_query + 1, dtype=torch.float32, device=scores.device)
        
        gains = torch.pow(2, labels) - 1
        ideal_discounts = 1 / torch.log2(ideal_ranks + 1)
        idcg = torch.sum(gains * ideal_discounts)
        
        if idcg == 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

        # Calculate current ranks based on model predictions
        sort_idx = torch.argsort(scores, descending=True)
        ranks = torch.zeros_like(sort_idx, dtype=torch.float32, device=scores.device)
        ranks[sort_idx] = torch.arange(1, num_docs_of_query + 1, dtype=torch.float32, device=scores.device)
        
        # Compute Delta NDCG for all pairs
        discounts = 1 / torch.log2(ranks + 1)
        delta_gains = torch.abs(gains.unsqueeze(1) - gains.unsqueeze(0)) # |[N, 1] - [1, N]|
        delta_discounts = torch.abs(discounts.unsqueeze(1) - discounts.unsqueeze(0))
        delta_ndcg = (delta_gains * delta_discounts) / idcg

        # Compute LambdaRank Loss: Delta NDCG * RankNet Loss
        # RankNet loss: -log(sigmoid(s_i - s_j))
        ranknet_loss = -torch.log(torch.sigmoid(scores_diff) + 1e-10)
        loss_matrix = delta_ndcg * ranknet_loss
        
        # Sum over valid pairs and average by number of positive pairs
        loss = torch.sum(loss_matrix * S_ij) / S_ij.sum()
        return loss

criterion = LambdaRankLoss()

# print number of parameters
num_params = sum(p.numel() for p in net.parameters())
print(f"Number of model parameters: {num_params}")

if args.checkpoint:
    checkpoint = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(checkpoint)
    print(f"Weights loaded from checkpoint: {args.checkpoint}")


def train(net):
    net.train()
    train_loss = 0.0
    num_valid_queries = 0
    doc_count = 0
    
    # Shuffle queries for the epoch
    qids = list(train_data_by_qid.keys())
    random.shuffle(qids)
    
    optimizer.zero_grad()
    
    for qid in qids:
        query_data = train_data_by_qid[qid]
        features = torch.from_numpy(np.array(query_data['features'])).float().to(device)
        labels = torch.from_numpy(np.array(query_data['labels'])).float().to(device)
        
        scores = net(features).squeeze()
        loss = criterion(scores, labels)
        
        if loss.item() > 0:
            loss.backward()
            train_loss += loss.item()
            num_valid_queries += 1
            
        # Gradient accumulation to simulate `batch_size` 
        doc_count += len(labels)
        if doc_count >= batch_size:
            optimizer.step()
            optimizer.zero_grad()
            doc_count = 0
            
    # Step optimizer for any remaining accumulated gradients
    if doc_count > 0:
        optimizer.step()
        optimizer.zero_grad()
        
    return train_loss / max(1, num_valid_queries)

def test(net, data_iter):
    net.eval()
    results = {}
    
    with torch.no_grad():
        for features, labels, qids in data_iter:
            scores = net(features).squeeze().data.cpu()
            
            # Handle edge case where batch size is 1 causing scalar squeeze
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)
                
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
    
    if args.task == 'train':
        print('Start training the model...')
        
        best_ndcg = float('-inf')
        best_model_state = None
        for epoch in range(num_epochs+1):
            # Train (skip actual training on epoch 0 for baseline)
            train_loss = train(net) if epoch > 0 else 0.0
            
            # Evaluate on validation and test sets
            val_metrics = test(net, val_reader_iter)
            test_metrics = test(net, test_reader_iter)
            print(f'epoch: {epoch}, train_loss: {(train_loss or 0.0):.6f}, val_map: {val_metrics["map"]:.6f}, val_ndcg: {val_metrics["ndcg"]:.6f}, test_map: {test_metrics["map"]:.6f}, test_ndcg: {test_metrics["ndcg"]:.6f}')            
            wandb.log({'loss/train_d_loss': train_loss,
                       'ranking_metrics/val_ndcg': val_metrics["ndcg"],
                       'ranking_metrics/val_map': val_metrics["map"],
                       'ranking_metrics/test_ndcg': test_metrics["ndcg"],
                       'ranking_metrics/test_map': test_metrics["map"],})
            
            # Track Best Model
            if val_metrics["ndcg"] > best_ndcg:
                best_ndcg = val_metrics["ndcg"]
                best_model_state = deepcopy(net.state_dict())

        final_model_save_path = os.path.join(project_root, 'discriminative', 'checkpoints', f'ltr.{dataset}.listwise.k{k}.{args.exp_name}.final.pt')
        best_model_save_path = os.path.join(project_root, 'discriminative', 'checkpoints', f'ltr.{dataset}.listwise.k{k}.{args.exp_name}.best.pt')

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
    results_save_path = os.path.join(project_root, 'discriminative', 'predictions', f'ltr.{dataset}.listwise.k{k}.{args.exp_name}.best.txt')
    with open(results_save_path, 'w') as f:
        f.write('qid true_label pred_label\n')
        for qid, values in test_metrics["results"].items():
            for true_label, pred_label in values:
                f.write(f'{qid} {true_label} {pred_label}\n')
    print('Results saved to {}'.format(results_save_path))

    wandb.finish()
