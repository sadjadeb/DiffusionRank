import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from utils import set_all_seeds, calculate_metrics, get_features_count
from model import DNN
from sklearn.preprocessing import QuantileTransformer
from copy import deepcopy
import argparse


# Set random seeds for reproducibility
seed = 42
set_all_seeds(seed)

# get k from command line arguments
parser = argparse.ArgumentParser(description='Train and test a pointwise neural network for Learning-to-Rank tasks')
parser.add_argument('--dataset', type=str, required=True, choices=['MQ2007', 'MQ2008', 'MSLR-WEB10K', 'MSLR-WEB30K'], help='Dataset to use for the experiment (required)')
parser.add_argument('--task', type=str, choices=['train', 'test'], help='Task to run: train or test') 
parser.add_argument('--k', type=float, default=1.0, help='Fraction k for the dataset')
parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging')
parser.add_argument('--exp_name', type=str, default=None, help='Experiment name for logging')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to the model checkpoint to load')
parser.add_argument('--save_best_by', type=str, default='ndcg', choices=['ndcg', 'loss'], help='Criterion to save the best model by')
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

features_count = get_features_count(args.dataset)
if args.dataset in ("MSLR-WEB10K", "MSLR-WEB30K", "Istella-S"):
    num_epochs = 2000
    threshold_of_neg = 1
else:
    num_epochs = 5000
    threshold_of_neg = 0

wandb.init(project=f"DiffusionRank_{dataset}", 
           name=f"disc_pointwise_k{k}" + (f"_{args.exp_name}" if args.exp_name else ""), 
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
    'save_best_by': args.save_best_by,
})

# Set data paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_root, 'data', dataset, 'by_fraction', 'Fold1', f'k{k}')


X_train = np.load(os.path.join(data_dir, 'X_num_train.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train.npy'))

X_val = np.load(os.path.join(data_dir, 'X_num_val.npy'))
y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
idx_val = np.load(os.path.join(data_dir, 'idx_val.npy'))

X_test = np.load(os.path.join(data_dir, 'X_num_test.npy'))
y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
idx_test = np.load(os.path.join(data_dir, 'idx_test.npy'))

# Binarize labels
threshold_of_neg = 1 if 'MSLR' in dataset else 0
bin_y_train = np.empty_like(y_train)
bin_y_val = np.empty_like(y_val)
bin_y_test = np.empty_like(y_test)
bin_y_train[y_train <= threshold_of_neg], bin_y_train[y_train > threshold_of_neg] = 0, 1
bin_y_val[y_val <= threshold_of_neg], bin_y_val[y_val > threshold_of_neg] = 0, 1
bin_y_test[y_test <= threshold_of_neg], bin_y_test[y_test > threshold_of_neg] = 0, 1

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

# Create a dataloader for the train, val, and test data
train_reader = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float().to(device), torch.from_numpy(bin_y_train).long().to(device))
train_reader_iter = torch.utils.data.DataLoader(train_reader, batch_size=batch_size, shuffle=True)

val_reader = torch.utils.data.TensorDataset(torch.from_numpy(X_val).float().to(device), torch.from_numpy(y_val).long(), torch.from_numpy(bin_y_val).long(), torch.from_numpy(idx_val).long())
val_reader_iter = torch.utils.data.DataLoader(val_reader, batch_size=batch_size, shuffle=False)

test_reader = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float().to(device), torch.from_numpy(y_test).long(), torch.from_numpy(bin_y_test).long(), torch.from_numpy(idx_test).long())
test_reader_iter = torch.utils.data.DataLoader(test_reader, batch_size=batch_size, shuffle=False)

# Create model, optimizer, and loss function
net = DNN(input_dim=features_count, approach='pointwise', num_hidden_nodes=num_hidden_nodes, dropout_rate=dropout_rate).to(device)
optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

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
    num_batches = 0
    
    for features, labels in train_reader_iter:
        out = net(features)
        loss = criterion(out, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        num_batches += 1
    
    return train_loss / num_batches


def test(net, data_iter):
    net.eval()
    results = {}
    avg_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for features, labels, bin_labels, qids in data_iter:
            raw_logits = net(features).data.cpu()
            
            # Take log softmax on the unnormalized probabilities to the logits
            logits = raw_logits - torch.logsumexp(raw_logits, dim=1, keepdim=True)

            loss = criterion(logits, bin_labels)
            avg_loss += loss.item()
            num_batches += 1

            row_cnt = len(qids)
            for i in range(row_cnt):
                qid = qids[i].item()
                if qid not in results:
                    results[qid] = []
                results[qid].append((labels[i], logits[i][1]))
        
        avg_loss /= num_batches

        avg_ndcg, avg_map = calculate_metrics(results)
        
        return {
            'map': avg_map,
            'ndcg': avg_ndcg,
            'loss': avg_loss,
            'results': results
        }



if __name__ == '__main__':
    print(f'Dataset: {dataset}')
    print(f'Number of learnable parameters: {net.parameter_count()}')
    
    if args.task == 'train':
        print('Start training the model...')
        
        best_val_loss = float('inf')
        best_ndcg = float('-inf')
        best_model_state = None
        for epoch in range(num_epochs+1):
            # Train (skip actual training on epoch 0 for baseline)
            train_loss = train(net) if epoch > 0 else None
            
            # Evaluate on validation and test sets
            val_metrics = test(net, val_reader_iter)
            test_metrics = test(net, test_reader_iter)
            print(f'epoch: {epoch}, train_loss: {(train_loss or 0.0):.6f}, val_loss: {val_metrics["loss"]:.6f}, val_map: {val_metrics["map"]:.6f}, val_ndcg: {val_metrics["ndcg"]:.6f}, test_loss: {test_metrics["loss"]:.6f}, test_map: {test_metrics["map"]:.6f}, test_ndcg: {test_metrics["ndcg"]:.6f}')
            wandb.log({'loss/train_d_loss': train_loss,
                       'ranking_metrics/val_ndcg': val_metrics["ndcg"],
                       'ranking_metrics/val_map': val_metrics["map"],
                       'loss/val_d_loss': val_metrics["loss"],
                       'ranking_metrics/test_ndcg': test_metrics["ndcg"],
                       'ranking_metrics/test_map': test_metrics["map"],
                       'loss/test_d_loss': test_metrics["loss"],})
            
            # Track Best Model
            if args.save_best_by == 'loss':
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    best_model_state = deepcopy(net.state_dict())
            elif args.save_best_by == 'ndcg':
                if val_metrics["ndcg"] > best_ndcg:
                    best_ndcg = val_metrics["ndcg"]
                    best_model_state = deepcopy(net.state_dict())
            else:
                raise ValueError(f"Unknown save_best_by criterion: {args.save_best_by}")

        final_model_save_path = os.path.join(project_root, 'discriminative', 'checkpoints', f'ltr.{dataset}.pointwise.k{k}.{args.exp_name}.final.pt')
        best_model_save_path = os.path.join(project_root, 'discriminative', 'checkpoints', f'ltr.{dataset}.pointwise.k{k}.{args.exp_name}.best.pt')

        # Save model
        torch.save(net.state_dict(), final_model_save_path)
        print(f'Final model saved to {final_model_save_path}')

        # Load best model before saving
        net.load_state_dict(best_model_state)
        torch.save(net.state_dict(), best_model_save_path)
        print(f'Best model saved to {best_model_save_path}')
    
    print('Evaluating the best model on test set...')
    test_metrics = test(net, test_reader_iter)
    print(f'Best Model Performance: test_loss: {test_metrics["loss"]}, test_map: {test_metrics["map"]}, test_ndcg: {test_metrics["ndcg"]}')
        
    # Save results
    results_save_path = os.path.join(project_root, 'discriminative', 'predictions', f'ltr.{dataset}.pointwise.k{k}.{args.exp_name}.best.txt')
    with open(results_save_path, 'w') as f:
        f.write('qid true_label pred_label\n')
        for qid, values in test_metrics["results"].items():
            for true_label, pred_label in values:
                f.write(f'{qid} {true_label} {pred_label}\n')
    print('Results saved to {}'.format(results_save_path))

    wandb.finish()
