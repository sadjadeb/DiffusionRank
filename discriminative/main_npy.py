import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import ndcg_score
from utils import set_all_seeds
from data_loader import DataLoaderTest
from model import DNN

seed = 42
set_all_seeds(seed)

approach = 'pointwise' # ['pointwise', 'pairwise']
dataset = 'MQ2007' # ['MQ2007', 'MQ2008', 'MSLR-Web10K', 'MSLR-Web30K']

# Set hyperparameters
device = torch.device("cuda:3")
features_count = 136 if 'MSLR' in dataset else 46
num_steps_per_epoch = 2048
num_epochs = 100
dropout_rate = 0.1
learning_rate = 1e-5
batch_size = 1024

# Set data paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_root, 'data', dataset, 'raw', 'Fold1')
train_data_file = os.path.join(data_dir, 'train.txt')
test_data_file = os.path.join(data_dir, 'test.txt')
model_output_file_path = f"ltr.{dataset}.{num_epochs}.pth"

wandb.init(project=f"ltr_npy_{dataset}", name=f"real1x+synthetic1x")
wandb.config.update({
    'features_count': features_count,
    'num_epochs': num_epochs,
    'dropout_rate': dropout_rate,
    'learning_rate': learning_rate,
})


net = DNN(input_dim=features_count, approach=approach, dropout_rate=dropout_rate).to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
if approach == 'pairwise':
    criterion = nn.CrossEntropyLoss()
elif approach == 'pointwise':
    criterion = nn.MSELoss()

test_reader = DataLoaderTest(test_data_file, batch_size=batch_size, features_count=features_count, device=device)
test_reader_iter = iter(test_reader)

# Start with the pointwise approach because we need the number of data points to find the EPOCH_STEPS, then overwrite the READER_TRAIN_ITER with the pairwise approach
X_train_real = np.load(os.path.join(project_root, 'data', dataset, 'npy', 'Fold1', 'X_num_train.npy'))
y_train_real = np.load(os.path.join(project_root, 'data', dataset, 'npy', 'Fold1', 'y_train.npy'))
X_train_synthetic = np.load(os.path.join(project_root, 'generative', 'experiments', dataset, 'exp_2024-11-04_18-38-43', 'X_num_train.npy'))
y_train_synthetic = np.load(os.path.join(project_root, 'generative', 'experiments', dataset, 'exp_2024-11-04_18-38-43', 'y_train.npy'))

synthetic_data_ratio = 1
new_length = int(X_train_real.shape[0] * synthetic_data_ratio)
random_indices = np.random.choice(X_train_synthetic.shape[0], size=new_length, replace=False)
X_train_synthetic = X_train_synthetic[random_indices]
y_train_synthetic = y_train_synthetic[random_indices]

X_train = np.concatenate((X_train_real, X_train_synthetic), axis=0)
y_train = np.concatenate((y_train_real, y_train_synthetic), axis=0)

# Create a dataloader for the training data using pytorch
train_reader = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float().to(device), torch.from_numpy(y_train).float().to(device))
train_reader_iter = torch.utils.data.DataLoader(train_reader, batch_size=batch_size, shuffle=True)
    

def train(net):
    net.train()
    train_loss = 0.0
    e_size = 0
    for features, labels in train_reader_iter:
        e_size += 1
        if approach == 'pointwise':
            labels = labels.view(-1, 1)
            out = net(features)
            loss = criterion(out, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    return train_loss / e_size


def test(net, epoch, train_loss):
    net.eval()
    with torch.no_grad():
        results = {}
        val_loss = 0.0
        val_size = 0
        for features, qids, labels, cnt in test_reader_iter:
            out = net(features).data.cpu()
            labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
            loss = criterion(out, labels_tensor)
            val_loss += loss.item()
            val_size += 1
            row_cnt = len(qids)
            for i in range(row_cnt):
                if qids[i] not in results:
                    results[qids[i]] = []
                results[qids[i]].append((labels[i], out[i][0]))
        val_loss /= val_size

        total_precision = 0
        total_ndcg = 0
        for qid, labels in results.items():
            # Extract true labels and predicted labels
            true_labels = [t[0] for t in labels]
            predicted_labels = [t[1] for t in labels]

            # Sort based on predicted labels in descending order to calculate P@10
            sorted_indices = sorted(range(len(predicted_labels)), key=lambda i: predicted_labels[i], reverse=True)
            top_10_indices = sorted_indices[:10]

            # Precision at 10
            relevant_at_10 = sum(1 for i in top_10_indices if true_labels[i] > 0)
            precision_at_10 = relevant_at_10 / 10

            # NDCG@10
            ndcg_at_10 = ndcg_score([true_labels], [predicted_labels], k=10)

            total_ndcg += ndcg_at_10
            total_precision += precision_at_10

        # Calculate averages
        avgp = total_precision / len(results)
        avgndcg = total_ndcg / len(results)
        
        print(f'epoch:{epoch}, loss: {train_loss}, val_loss: {val_loss}, avgp: {avgp}, avgndcg: {avgndcg}')
        
        return avgp, avgndcg, val_loss


if __name__ == '__main__':
    print('Approach: {}'.format(approach))
    print('Dataset: {}'.format(dataset))
    print('Number of learnable parameters: {}'.format(net.parameter_count()))

    test(net, 0, 'n/a')
    for epoch in range(num_epochs):
        train_loss = train(net)    
        avgp, avgndcg, val_loss = test(net, epoch + 1, str(train_loss))
        
        wandb.log({'train_loss': train_loss, 'avgndcg': avgndcg, 'avgp': avgp, 'val_loss': val_loss})

    # save model
    torch.save(net.state_dict(), model_output_file_path)
    print('Model saved to {}'.format(model_output_file_path))
    wandb.finish()
