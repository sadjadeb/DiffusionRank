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


seed = 42
set_all_seeds(seed)

# get k from command line arguments
parser = argparse.ArgumentParser(description='Train and test a LambdaRank-NDCG neural network for Learning-to-Rank tasks')
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
           name=f"disc_lambdarank_ndcg_k{k}" + (f"_{args.exp_name}" if args.exp_name else ""), 
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

# Organize training data by query ID for efficient pair sampling and NDCG computation
train_data_by_qid = {}
for i in range(len(X_train)):
    qid = idx_train[i]
    if qid not in train_data_by_qid:
        train_data_by_qid[qid] = {'features': [], 'labels': []}
    train_data_by_qid[qid]['features'].append(X_train[i])
    train_data_by_qid[qid]['labels'].append(y_train[i])

# Filter out queries with only one document (can't form pairs)
num_queries_before = len(train_data_by_qid)
train_data_by_qid = {qid: data for qid, data in train_data_by_qid.items() if len(data['labels']) > 1}
num_queries_after = len(train_data_by_qid)
num_filtered = num_queries_before - num_queries_after
print(f"Organized training data into {num_queries_after} queries with multiple documents (filtered out {num_filtered} queries with only one document)")

# Create a dataloader for the validation data using pytorch
val_reader = torch.utils.data.TensorDataset(torch.from_numpy(X_val).float().to(device), torch.from_numpy(y_val).long(), torch.from_numpy(idx_val).long())
val_reader_iter = torch.utils.data.DataLoader(val_reader, batch_size=batch_size, shuffle=False)
# Create a dataloader for the test data using pytorch
test_reader = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float().to(device), torch.from_numpy(y_test).long(), torch.from_numpy(idx_test).long())
test_reader_iter = torch.utils.data.DataLoader(test_reader, batch_size=batch_size, shuffle=False)

# Create model and optimizer
net = DNN(input_dim=features_count, approach='pairwise', num_hidden_nodes=num_hidden_nodes, dropout_rate=dropout_rate).to(device)
optimizer = optim.AdamW(net.parameters(), lr=learning_rate)

# Cache for lambda weights
# Structure: {qid: {(idx_i, idx_j): lambda_weight, ...}, ...}
# Computed once every 50 epochs to avoid expensive forward passes
lambda_weights_cache = {}


class LambdaRankLoss(nn.Module):
    """
    LambdaRank loss function for pairwise learning to rank.
    Similar to RankNet but weighted by NDCG impact (lambda).
    Uses full ranking NDCG (no cutoff) as in standard LambdaRank.
    
    Reference: Burges et al. "Learning to Rank with Nonsmooth Cost Functions" (NIPS 2006)
    """
    def __init__(self):
        super(LambdaRankLoss, self).__init__()
    
    def forward(self, scores_i, scores_j, lambda_weights):
        """
        Args:
            scores_i: scores for documents that should be ranked higher (shape: [batch_size])
            scores_j: scores for documents that should be ranked lower (shape: [batch_size])
            lambda_weights: NDCG-based weights for each pair (shape: [batch_size])
        Returns:
            loss: LambdaRank loss
        """
        # Compute the difference
        diff = scores_i - scores_j
        # RankNet loss: -log(sigmoid(diff))
        ranknet_loss = -torch.log(torch.sigmoid(diff) + 1e-10)
        # Weight by lambda (NDCG impact)
        weighted_loss = (lambda_weights * ranknet_loss).mean()
        return weighted_loss


criterion = LambdaRankLoss()

# print number of parameters
num_params = sum(p.numel() for p in net.parameters())
print(f"Number of model parameters: {num_params}")

if args.checkpoint:
    checkpoint = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(checkpoint)
    print(f"Weights loaded from checkpoint: {args.checkpoint}")


def compute_lambda_weight_with_scores(scores, labels, idx_i, idx_j):
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


def compute_and_cache_lambda_weights(qid_data, device):
    """
    Pre-compute lambda weights for all document pairs in all queries.
    This is called once every 50 epochs to populate the lambda_weights_cache.
    
    Args:
        qid_data: Dictionary mapping query IDs to document features and labels
        device: PyTorch device
    
    Returns:
        Dictionary mapping (qid, idx_i, idx_j) to lambda weights
    """
    global lambda_weights_cache
    lambda_weights_cache = {}
    
    net.eval()
    print(f"  Pre-computing lambda weights for {len(qid_data)} queries...")
    
    with torch.no_grad():
        for qid, query_data in qid_data.items():
            n_docs = len(query_data['labels'])
            if n_docs < 2:
                continue
            
            # Get model scores for all documents in this query
            query_features = torch.from_numpy(np.array(query_data['features'])).float().to(device)
            query_scores = net(query_features).squeeze().cpu().numpy()
            query_labels = np.array(query_data['labels'])
            
            # Initialize cache for this query
            if qid not in lambda_weights_cache:
                lambda_weights_cache[qid] = {}
            
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
                    lambda_weight = compute_lambda_weight_with_scores(
                        query_scores, query_labels, actual_idx_i, actual_idx_j
                    )
                    lambda_weight = max(lambda_weight, 1e-10)
                    
                    # Store in cache (store both orders for easy lookup)
                    lambda_weights_cache[qid][(actual_idx_i, actual_idx_j)] = lambda_weight
                    lambda_weights_cache[qid][(actual_idx_j, actual_idx_i)] = lambda_weight
    
    print(f"  Lambda weights cached for {len(lambda_weights_cache)} queries")


def generate_pairs_with_lambdas(qid_data, batch_size, device, use_lambda_weights=True):
    """
    Generate pairs of documents with lambda weights for LambdaRank training.
    Lambda weights are looked up from the pre-computed cache (no forward passes needed).
    
    Args:
        qid_data: Dictionary mapping query IDs to document features and labels
        batch_size: Number of pairs to generate
        device: PyTorch device
        use_lambda_weights: If True, use cached NDCG-based lambda weights. If False, use equal weights (RankNet mode)
    """
    global lambda_weights_cache
    
    if use_lambda_weights and not lambda_weights_cache:
        raise ValueError("lambda_weights_cache is empty. Must compute and cache lambda weights first.")
    
    pairs_i = []
    pairs_j = []
    lambdas = []
    
    qids = list(qid_data.keys())
    attempts = 0
    max_attempts = batch_size * 10
    
    while len(pairs_i) < batch_size and attempts < max_attempts:
        attempts += 1
        qid = random.choice(qids)
        query_data = qid_data[qid]
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
            if qid in lambda_weights_cache and (idx_i, idx_j) in lambda_weights_cache[qid]:
                lambda_weight = lambda_weights_cache[qid][(idx_i, idx_j)]
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
        
    if use_lambda_weights:
        print(f"Lambda weights stats - mean: {np.mean(lambdas)}, median: {np.median(lambdas)}, max: {np.max(lambdas)}, min: {np.min(lambdas)}")
    
    return (torch.from_numpy(np.array(pairs_i)).float().to(device), 
            torch.from_numpy(np.array(pairs_j)).float().to(device),
            torch.from_numpy(np.array(lambdas)).float().to(device))


def train(net, epoch):
    global lambda_weights_cache
    
    net.train()
    train_loss = 0.0
    num_batches = 0
    
    if epoch <= 50:
        use_lambda_weights = False
    else:
        # Check if we need to recompute lambda weights (at epochs 51, 101, 151, ...)
        if (epoch - 51) % 50 == 0:  # epochs 51, 101, 151, ...
            print(f"  Epoch {epoch}: Computing and caching lambda weights (will be used for epochs {epoch}-{epoch+49})")
            compute_and_cache_lambda_weights(train_data_by_qid, device)
        
        use_lambda_weights = True
    
    # Print mode info on first epoch
    if epoch == 1:
        print(f"  Using RankNet mode (equal weights) for first 50 epochs")
    
    num_batches_per_epoch = max(1, len(X_train) // batch_size)
    for _ in range(num_batches_per_epoch):
        features_i, features_j, lambda_weights = generate_pairs_with_lambdas(
            train_data_by_qid, batch_size, device, use_lambda_weights=use_lambda_weights
        )
        
        # Get scores for both documents
        scores_i = net(features_i).squeeze()
        scores_j = net(features_j).squeeze()
        
        # Compute LambdaRank loss (RankNet weighted by lambda)
        loss = criterion(scores_i, scores_j, lambda_weights)
        
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
    
    if args.task == 'train':
        print('Start training the model...')
        best_ndcg = float('-inf')
        best_model_state = None
        for epoch in range(num_epochs+1):
            # Train (skip actual training on epoch 0 for baseline)
            train_loss = train(net, epoch) if epoch > 0 else None
            
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

        final_model_save_path = os.path.join(project_root, 'discriminative', 'checkpoints', f'ltr.{dataset}.lambdarank_ndcg.k{k}.{args.exp_name}.final.pt')
        best_model_save_path = os.path.join(project_root, 'discriminative', 'checkpoints', f'ltr.{dataset}.lambdarank_ndcg.k{k}.{args.exp_name}.best.pt')

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
    results_save_path = os.path.join(project_root, 'discriminative', 'predictions', f'ltr.{dataset}.lambdarank_ndcg.k{k}.{args.exp_name}.best.txt')
    with open(results_save_path, 'w') as f:
        f.write('qid true_label pred_label\n')
        for qid, values in test_metrics["results"].items():
            for true_label, pred_label in values:
                f.write(f'{qid} {true_label} {pred_label}\n')
    print('Results saved to {}'.format(results_save_path))

    wandb.finish()
