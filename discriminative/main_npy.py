import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from utils import set_all_seeds, calculate_metrics
from model import DNN

seed = 42
set_all_seeds(seed)

approach = 'pointwise' # ['pointwise', 'pairwise']
dataset = 'MQ2007' # ['MQ2007', 'MQ2008', 'MSLR-WEB10K', 'MSLR-WEB30K']
k = 1.0

# Set hyperparameters
device = torch.device("cuda:0")
features_count = 136 if 'MSLR' in dataset else 46
num_steps_per_epoch = 2048
num_epochs = 100
dropout_rate = 0.2 if 'MSLR' in dataset else 0.1
learning_rate = 5e-4 if 'MSLR' in dataset else 1e-5
num_hidden_nodes = 256 if 'MSLR' in dataset else 128
batch_size = 1024

wandb.init(project=f"ltr_npy_{dataset}", name=f"exp_k{k}_{approach}")
wandb.config.update({
    'features_count': features_count,
    'num_epochs': num_epochs,
    'dropout_rate': dropout_rate,
    'learning_rate': learning_rate,
})

# Set data paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

X_train = np.load(os.path.join(project_root, 'data', dataset, 'by_fraction', f'k{k}', 'X_num_train.npy'))
y_train = np.load(os.path.join(project_root, 'data', dataset, 'by_fraction', f'k{k}', 'y_train.npy'))
# Create a dataloader for the training data using pytorch
train_reader = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float().to(device), torch.from_numpy(y_train).float().to(device))
train_reader_iter = torch.utils.data.DataLoader(train_reader, batch_size=batch_size, shuffle=True)

X_test = np.load(os.path.join(project_root, 'data', dataset, 'by_fraction', f'k{k}', 'X_num_test.npy'))
y_test = np.load(os.path.join(project_root, 'data', dataset, 'by_fraction', f'k{k}', 'y_test.npy'))
idx_test = np.load(os.path.join(project_root, 'data', dataset, 'by_fraction', f'k{k}', 'idx_test.npy'))
# Create a dataloader for the test data using pytorch
test_reader = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float().to(device), torch.from_numpy(y_test).float(), torch.from_numpy(idx_test).long())
test_reader_iter = torch.utils.data.DataLoader(test_reader, batch_size=batch_size, shuffle=False)

# Create model, optimizer, and loss function
net = DNN(input_dim=features_count, num_hidden_layers=4, num_hidden_nodes=num_hidden_nodes, approach=approach, dropout_rate=dropout_rate).to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
if approach == 'pairwise':
    criterion = nn.CrossEntropyLoss()
elif approach == 'pointwise':
    criterion = nn.MSELoss()
    

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
        for features, labels, qids in test_reader_iter:
            out = net(features).data.cpu()
            labels_tensor = labels.view(-1, 1)
            loss = criterion(out, labels_tensor)
            val_loss += loss.item()
            val_size += 1
            row_cnt = len(qids)
            for i in range(row_cnt):
                qid = qids[i].item()
                if qid not in results:
                    results[qid] = []
                results[qid].append((labels[i], out[i][0]))
        val_loss /= val_size

        avgndcg, avgp = calculate_metrics(results)
        
        print(f'epoch:{epoch}, loss: {train_loss}, val_loss: {val_loss}, avgp: {avgp}, avgndcg: {avgndcg}')
        
        return avgp, avgndcg, val_loss, results


if __name__ == '__main__':
    print('Approach: {}'.format(approach))
    print('Dataset: {}'.format(dataset))
    print('Number of learnable parameters: {}'.format(net.parameter_count()))
    
    best_val_loss = float('inf')
    best_model_state = None
    best_results = None

    test(net, 0, 'n/a')
    for epoch in range(num_epochs):
        train_loss = train(net)    
        avgp, avgndcg, val_loss, results = test(net, epoch + 1, str(train_loss))
        
        wandb.log({'train_loss': train_loss, 'avgndcg': avgndcg, 'avgp': avgp, 'val_loss': val_loss})
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = net.state_dict().copy()
            best_results = results
            
    final_model_save_path = os.path.join(project_root, 'discriminative', 'experiments', f'ltr.{dataset}.k{k}.final.pt')
    best_model_save_path = os.path.join(project_root, 'discriminative', 'experiments', f'ltr.{dataset}.k{k}.best.pt')
    
        
    # Save model
    torch.save(net.state_dict(), final_model_save_path)
    print('Final model saved to {}'.format(final_model_save_path))

    # Load best model before saving
    if best_model_state is not None:
        net.load_state_dict(best_model_state)
        torch.save(net.state_dict(), best_model_save_path)
        print('Best model saved to {}'.format(best_model_save_path))
        
    # Save results
    results_save_path = os.path.join(project_root, 'discriminative', 'experiments', f'ltr.{dataset}.k{k}.best.results.txt')
    with open(results_save_path, 'w') as f:
        f.write('qid true_label pred_label\n')
        for qid, values in best_results.items():
            for true_label, pred_label in values:
                f.write(f'{qid} {true_label} {pred_label}\n')
    print('Results saved to {}'.format(results_save_path))

    wandb.finish()
