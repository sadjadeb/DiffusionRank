import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from utils import set_all_seeds, calculate_metrics
from model import DNN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import QuantileTransformer
from copy import deepcopy
import argparse


seed = 42
set_all_seeds(seed)

# get k from command line arguments
parser = argparse.ArgumentParser(description='Run LTR Classifier Experiment')
parser.add_argument('--dataset', type=str, choices=['MQ2007', 'MQ2008', 'MSLR-WEB10K', 'MSLR-WEB30K'], help='Dataset to use for the experiment')
parser.add_argument('--task', type=str, choices=['train', 'test'], help='Task to run: train or test') 
parser.add_argument('--k', type=float, default=1.0, help='Fraction k for the dataset')
parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to the model checkpoint to load')
parser.add_argument('--save_best_by', type=str, default='loss', choices=['ndcg', 'loss'], help='Criterion to save the best model by')
parser.add_argument('--num_hidden_nodes', type=int, help='Number of hidden nodes in the model')
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
data_normalization = 'quantile'  # ['quantile', None]
num_epochs = 5000
dropout_rate = 0.1
learning_rate = 5e-6
num_hidden_nodes = args.num_hidden_nodes
batch_size = 4096

wandb.init(project=f"ltr_npy_{dataset}_classifier", 
           name=f"exp_k{k}", 
           mode='disabled' if args.no_wandb else 'online')
wandb.config.update({
    'features_count': features_count,
    'data_normalization': data_normalization,
    'num_epochs': num_epochs,
    'dropout_rate': dropout_rate,
    'learning_rate': learning_rate,
    'num_hidden_nodes': num_hidden_nodes,
    'batch_size': batch_size,
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
y_train[y_train <= threshold_of_neg], y_train[y_train > threshold_of_neg] = 0, 1
y_val[y_val <= threshold_of_neg], y_val[y_val > threshold_of_neg] = 0, 1
y_test[y_test <= threshold_of_neg], y_test[y_test > threshold_of_neg] = 0, 1

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

# Create a dataloader for the training data using pytorch
train_reader = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float().to(device), torch.from_numpy(y_train).long().to(device))
train_reader_iter = torch.utils.data.DataLoader(train_reader, batch_size=batch_size, shuffle=True)
# Create a dataloader for the validation data using pytorch
val_reader = torch.utils.data.TensorDataset(torch.from_numpy(X_val).float().to(device), torch.from_numpy(y_val).long(), torch.from_numpy(idx_val).long())
val_reader_iter = torch.utils.data.DataLoader(val_reader, batch_size=batch_size, shuffle=False)
# Create a dataloader for the test data using pytorch
test_reader = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float().to(device), torch.from_numpy(y_test).long(), torch.from_numpy(idx_test).long())
test_reader_iter = torch.utils.data.DataLoader(test_reader, batch_size=batch_size, shuffle=False)

# Create model, optimizer, and loss function
net = DNN(input_dim=features_count, approach='pointwise_classifier', num_hidden_nodes=num_hidden_nodes, dropout_rate=dropout_rate).to(device)
optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

if args.checkpoint:
    checkpoint = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(checkpoint)
    print(f"Weights loaded from checkpoint: {args.checkpoint}")


def train(net):
    net.train()
    train_loss = 0.0
    e_size = 0
    for features, labels in train_reader_iter:
        e_size += 1
        out = net(features)
        loss = criterion(out, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    return train_loss / e_size


def test(net, data_iter):
    net.eval()
    with torch.no_grad():
        results = {}
        val_loss = 0.0
        val_acc = 0.0
        val_size = 0
        for features, labels, qids in data_iter:
            raw_logits = net(features).data.cpu()
            
            # Take log softmax on the unnormalized probabilities to the logits
            logits = raw_logits - torch.logsumexp(raw_logits, dim=1, keepdim=True)

            loss = criterion(logits, labels)
            val_loss += loss.item()
            val_size += 1

            _, preds = torch.max(logits, 1)
            acc = accuracy_score(labels.numpy(), preds.numpy())
            val_acc += acc
            
            row_cnt = len(qids)
            for i in range(row_cnt):
                qid = qids[i].item()
                if qid not in results:
                    results[qid] = []
                results[qid].append((labels[i], logits[i][1]))
        val_loss /= val_size
        val_acc /= val_size

        avgndcg, avgp = calculate_metrics(results)
        
        return avgp, avgndcg, val_loss, val_acc, results


if __name__ == '__main__':
    print('Dataset: {}'.format(dataset))
    print('Number of learnable parameters: {}'.format(net.parameter_count()))
    
    if args.task == 'train':
        print('Training the model...')
        avgp, avgndcg, val_loss, val_acc, _ = test(net, val_reader_iter)
        print(f'epoch: 0, train_loss: None, val_loss: {val_loss}, p: {avgp}, ndcg: {avgndcg}, acc: {val_acc}')
        wandb.log({'train_loss': None, 'avgndcg': avgndcg, 'avgp': avgp, 'val_loss': val_loss, 'val_acc': val_acc})
        
        best_val_loss = float('inf')
        best_ndcg = float('-inf')
        best_model_state = None
        for epoch in range(num_epochs):
            train_loss = train(net)
            
            avgp, avgndcg, val_loss, val_acc, _ = test(net, val_reader_iter)
            print(f'epoch:{epoch+1}, train_loss: {train_loss}, val_loss: {val_loss}, p: {avgp}, ndcg: {avgndcg}, acc: {val_acc}')
            wandb.log({'train_loss': train_loss, 'avgndcg': avgndcg, 'avgp': avgp, 'val_loss': val_loss, 'val_acc': val_acc})
            
            # Save best model
            if args.save_best_by == 'loss':
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = deepcopy(net.state_dict())
            elif args.save_best_by == 'ndcg':
                if avgndcg > best_ndcg:
                    best_ndcg = avgndcg
                    best_model_state = deepcopy(net.state_dict())
            else:
                raise ValueError(f"Unknown save_best_by criterion: {args.save_best_by}")

        final_model_save_path = os.path.join(project_root, 'discriminative', 'checkpoints', f'ltr.{dataset}.classifier.final.pt')
        best_model_save_path = os.path.join(project_root, 'discriminative', 'checkpoints', f'ltr.{dataset}.classifier.best.pt')
            
        # Save model
        torch.save(net.state_dict(), final_model_save_path)
        print('Final model saved to {}'.format(final_model_save_path))

        # Load best model before saving
        net.load_state_dict(best_model_state)
        torch.save(net.state_dict(), best_model_save_path)
        print('Best model saved to {}'.format(best_model_save_path))
    
    print('Evaluating on test set...')
    avgp, avgndcg, test_loss, test_acc, test_results = test(net, test_reader_iter)
    print(f'Test Loss: {test_loss}, Test P: {avgp}, Test NDCG: {avgndcg}, Test Acc: {test_acc}')
        
    # Save results
    results_save_path = os.path.join(project_root, 'discriminative', 'predictions', f'ltr.{dataset}.classifier.best.txt')
    with open(results_save_path, 'w') as f:
        f.write('qid true_label pred_label\n')
        for qid, values in test_results.items():
            for true_label, pred_label in values:
                f.write(f'{qid} {true_label} {pred_label}\n')
    print('Results saved to {}'.format(results_save_path))

    wandb.finish()
