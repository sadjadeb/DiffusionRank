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


seed = 42
set_all_seeds(seed)

dataset = 'MQ2007' # ['MQ2007', 'MQ2008', 'MSLR-WEB10K', 'MSLR-WEB30K']
k = 1.0

# Set hyperparameters
device = torch.device("cuda:0")
features_count = 136 if 'MSLR' in dataset else 46
data_normalization = 'quantile'  # ['quantile', None]
num_epochs = 5000
dropout_rate = 0.1
learning_rate = 5e-6
num_hidden_nodes = 256 if 'MSLR' in dataset else 128
batch_size = 4096

features_count += 1 # Add 1 for time feature

wandb.init(project=f"ltr_npy_{dataset}_classifier", name=f"exp_perturbed_2dim")
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

X_train = np.load(os.path.join(project_root, 'data', dataset, 'by_fraction', f'k{k}', 'X_num_train.npy'))
y_train = np.load(os.path.join(project_root, 'data', dataset, 'by_fraction', f'k{k}', 'y_train.npy'))
# Replace all labels greater than 1 with 1
y_train[y_train > 1] = 1

X_val = np.load(os.path.join(project_root, 'data', dataset, 'by_fraction', f'k{k}', 'X_num_val.npy'))
y_val = np.load(os.path.join(project_root, 'data', dataset, 'by_fraction', f'k{k}', 'y_val.npy'))
idx_val = np.load(os.path.join(project_root, 'data', dataset, 'by_fraction', f'k{k}', 'idx_val.npy'))
# Replace all labels greater than 1 with 1
y_val[y_val > 1] = 1

X_test = np.load(os.path.join(project_root, 'data', dataset, 'by_fraction', f'k{k}', 'X_num_test.npy'))
y_test = np.load(os.path.join(project_root, 'data', dataset, 'by_fraction', f'k{k}', 'y_test.npy'))
idx_test = np.load(os.path.join(project_root, 'data', dataset, 'by_fraction', f'k{k}', 'idx_test.npy'))
# Replace all labels greater than 1 with 1
y_test[y_test > 1] = 1

if data_normalization == 'quantile':
    # Apply QuantileTransformer
    normalizer = QuantileTransformer(
                output_distribution='normal',
                n_quantiles=max(min(X_train.shape[0] // 30, 1000), 10),
                subsample=int(1e9),
                random_state=seed,
            )
    X_train = normalizer.fit_transform(X_train)
    X_val = normalizer.transform(X_val)
    X_test = normalizer.transform(X_test)

# Convert to torch tensors
X_train = torch.from_numpy(X_train).float().to(device)
y_train = torch.from_numpy(y_train).to(device)

# add zeros to the val and test sets for the time feature
X_val = np.concatenate((X_val, np.zeros((X_val.shape[0], 1))), axis=1)
X_test = np.concatenate((X_test, np.zeros((X_test.shape[0], 1))), axis=1)

# Create a dataloader for the validation and test data using pytorch
val_reader = torch.utils.data.TensorDataset(torch.from_numpy(X_val).float().to(device), torch.from_numpy(y_val).long(), torch.from_numpy(idx_val).long())
val_reader_iter = torch.utils.data.DataLoader(val_reader, batch_size=batch_size, shuffle=False)
test_reader = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float().to(device), torch.from_numpy(y_test).long(), torch.from_numpy(idx_test).long())
test_reader_iter = torch.utils.data.DataLoader(test_reader, batch_size=batch_size, shuffle=False)

# Create model, optimizer, and loss function
net = DNN(input_dim=features_count, num_hidden_layers=4, num_hidden_nodes=num_hidden_nodes, approach='classifier', dropout_rate=dropout_rate).to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


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


def train(net, train_reader_iter):
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


def evaluate(net, data_iter):
    net.eval()
    with torch.no_grad():
        results = {}
        val_loss = 0.0
        val_acc = 0.0
        val_size = 0
        for features, labels, qids in data_iter:
            out = net(features).data.cpu()
            
            loss = criterion(out, labels)
            val_loss += loss.item()
            val_size += 1
            
            _, preds = torch.max(out, 1)
            acc = accuracy_score(labels.numpy(), preds.numpy())
            val_acc += acc
            
            row_cnt = len(qids)
            for i in range(row_cnt):
                qid = qids[i].item()
                if qid not in results:
                    results[qid] = []
                results[qid].append((labels[i], out[i][1]))
        val_loss /= val_size
        val_acc /= val_size

        avgndcg, avgp = calculate_metrics(results)
        
        return avgp, avgndcg, val_loss, val_acc, results


if __name__ == '__main__':
    print('Approach: Classifier')
    print('Dataset: {}'.format(dataset))
    print('Number of learnable parameters: {}'.format(net.parameter_count()))
    
    avgp, avgndcg, val_loss, val_acc, _ = evaluate(net, val_reader_iter)
    print(f'epoch: 0, train_loss: None, val_loss: {val_loss}, p: {avgp}, ndcg: {avgndcg}, acc: {val_acc}')
    wandb.log({'train_loss': None, 'avgndcg': avgndcg, 'avgp': avgp, 'val_loss': val_loss, 'val_acc': val_acc})
    
    
    # Beta schedule (e.g., linear from 1e-5 1e-3 over 50 steps)
    T = 50
    beta_schedule = torch.linspace(1e-5, 1e-3, T).to(device)

    best_val_loss = float('inf')
    best_model_state = None
    for epoch in range(num_epochs):
        # Sample timestep t per sample in batch
        t = torch.randint(0, T, (X_train.size(0),), device=device)
        norm_t = t.float() / (T - 1)
        
        # Perturb inputs
        x_noisy, _ = forward_diffusion_step(X_train, t, beta_schedule)
        
        # Concatenate with time feature
        x_noisy = torch.cat((x_noisy, norm_t.view(-1, 1)), dim=1)
        
        train_reader = torch.utils.data.TensorDataset(x_noisy, y_train)
        train_reader_iter = torch.utils.data.DataLoader(train_reader, batch_size=batch_size, shuffle=True)
    
        train_loss = train(net, train_reader_iter)
        
        avgp, avgndcg, val_loss, val_acc, _ = evaluate(net, val_reader_iter)
        print(f'epoch:{epoch+1}, train_loss: {train_loss}, val_loss: {val_loss}, p: {avgp}, ndcg: {avgndcg}, acc: {val_acc}')
        wandb.log({'train_loss': train_loss, 'avgndcg': avgndcg, 'avgp': avgp, 'val_loss': val_loss, 'val_acc': val_acc})
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = net.state_dict().copy()


    final_model_save_path = os.path.join(project_root, 'discriminative', 'experiments', f'ltr.{dataset}.classifier.2dim.quantile.perturbed.final.pt')
    best_model_save_path = os.path.join(project_root, 'discriminative', 'experiments', f'ltr.{dataset}.classifier.2dim.quantile.perturbed.best.pt')
        
    # Save model
    torch.save(net.state_dict(), final_model_save_path)
    print('Final model saved to {}'.format(final_model_save_path))

    # Load best model before saving
    net.load_state_dict(best_model_state)
    torch.save(net.state_dict(), best_model_save_path)
    print('Best model saved to {}'.format(best_model_save_path))
    
    print('Evaluating on test set...')
    avgp, avgndcg, test_loss, test_acc, test_results = evaluate(net, test_reader_iter)
    print(f'Test Loss: {test_loss}, Test P: {avgp}, Test NDCG: {avgndcg}, Test Acc: {test_acc}')
        
    # Save results
    results_save_path = os.path.join(project_root, 'discriminative', 'experiments', f'ltr.{dataset}.classifier.2dim.quantile.perturbed.best.results.txt')
    with open(results_save_path, 'w') as f:
        f.write('qid true_label pred_label\n')
        for qid, values in test_results.items():
            for true_label, pred_label in values:
                f.write(f'{qid} {true_label} {pred_label}\n')
    print('Results saved to {}'.format(results_save_path))

    wandb.finish()
