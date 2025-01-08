import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import wandb
from argparse import ArgumentParser
from sklearn.metrics import ndcg_score

import generative.lib as lib
from generative.tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from generative.tab_ddpm.modules import MLPDiffusion
from generative.scripts.utils_train import make_dataset
from discriminative.data_loader import DataLoaderTrain, DataLoaderTest
from discriminative.model import DNN
from utils import set_all_seeds, EarlyStopping


seed = 42
set_all_seeds(seed)

parser = ArgumentParser()
parser.add_argument('--p', type=float)
parser.add_argument('--k', type=float)
parser.add_argument('--sub', type=int, choices=[1, 2], help="When sub=2, it uses the generative model that partially trained with unlabeled data.")
args = parser.parse_args()

p_value = args.p
k_value = args.k
sub_exp_no = args.sub

approach = 'pointwise'  # ['pointwise', 'pairwise']
dataset = 'MQ2007'  # ['MQ2007', 'MQ2008', 'MSLR-Web10K', 'MSLR-Web30K']

# Set hyperparameters
device = torch.device("cuda:3")
features_count = 136 if 'MSLR' in dataset else 46
num_epochs = 300
dropout_rate = 0.2
learning_rate = 2e-5
batch_size = 1024
early_stopping_patience = 15
early_stopping_min_delta = 0.001

# Set data paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_root, 'data', dataset, 'raw', 'Fold1')
train_data_file = os.path.join(data_dir, 'train.txt')
test_data_file = os.path.join(data_dir, 'test.txt')
model_output_file_path = os.path.join('experiments', f"sub{sub_exp_no}", f"ltr.{dataset}.p{p_value}.k{k_value}.pt")

num_synthetic_data_in_batch = int(batch_size * p_value)
num_real_data_in_batch = batch_size - num_synthetic_data_in_batch
print(f"Using {num_synthetic_data_in_batch} synthetic data points and {num_real_data_in_batch} real data points in whole batch with size {batch_size}")

if sub_exp_no == 1:
    generative_experiment_id = f"exp_k{k_value}"
elif sub_exp_no == 2:
    generative_experiment_id = f"exp_k{k_value}_unlabeled"
generative_config_path = os.path.join('..', 'generative', 'experiments', 'MQ2007', generative_experiment_id, 'config.toml')

wandb.init(project=f"ltr_by_synthetic_{dataset}_sub{sub_exp_no}_{approach}", name=f"exp_p{p_value}_k{k_value}", group=f"{k_value}")
wandb.config.update({
    "p": p_value,
    "k": k_value,
    "num_epochs": num_epochs,
    "dropout_rate": dropout_rate,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "early_stopping_patience": early_stopping_patience,
    "early_stopping_min_delta": early_stopping_min_delta,
    "num_synthetic_data_in_batch": num_synthetic_data_in_batch,
    "num_real_data_in_batch": num_real_data_in_batch,
    "generative_config_path": generative_config_path,
})


raw_config = lib.load_config(generative_config_path)

T_dict = raw_config['train']['T']
T = lib.Transformations(**T_dict)
D = make_dataset(
    os.path.join(raw_config['real_data_path']),
    T,
    num_classes=raw_config['model_params']['num_classes'],
    is_y_cond=raw_config['model_params']['is_y_cond'],
)
_, empirical_class_dist = torch.unique(torch.from_numpy(D.y['train']), return_counts=True)

K = np.array(D.get_category_sizes('train'))
if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
    K = np.array([0])

num_numerical_features_ = D.X_num['train'].shape[1] if D.X_num is not None else 0
d_in = np.sum(K) + num_numerical_features_
raw_config['model_params']['d_in'] = int(d_in)

model = MLPDiffusion(**raw_config['model_params'])

model_path = os.path.join('..', 'generative', raw_config['parent_dir'], 'model.pt')
model.load_state_dict(torch.load(model_path, map_location="cpu"))

diffusion = GaussianMultinomialDiffusion(
    **raw_config['diffusion_params'],
    num_classes=K,
    num_numerical_features=num_numerical_features_,
    denoise_fn=model,
    device=device,
)

diffusion.to(device)
diffusion.eval()


def generate_synthetic_data(batch_size=256, num_samples=0, num_numerical_features=features_count + 1):
    x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False,verbose=False)
    X_gen, y_gen = x_gen.numpy(), y_gen.numpy()

    num_numerical_features = num_numerical_features + 1
    X_num = D.num_transform.inverse_transform(X_gen[:, :num_numerical_features])
    X_num = X_num[:, :num_numerical_features]

    y_gen = X_num[:, 0]
    X_num = X_num[:, 1:]

    # cast to torch tensors
    X_num = torch.tensor(X_num, dtype=torch.float).to(device)
    y_gen = torch.tensor(y_gen, dtype=torch.float).to(device)

    return X_num, y_gen


def mixed_dataloader(real_dataloader, num_generated_samples, epoch_steps, device):
    """
    Create a dataloader that mixes real and synthetic data in each batch.
    Args:
        real_dataloader: PyTorch DataLoader for real data.
        num_generated_samples: Number of synthetic samples to generate.
        diffusion_model: Model that generates synthetic data.
        device: Device that the data should be sent to.
    Yields:
        Mixed batches of real and synthetic data.
    """
    if num_generated_samples == 0:
        for X_real, y_real in real_dataloader:
            yield X_real.to(device), y_real.to(device)
    elif real_dataloader is None:
        for _ in range(epoch_steps):
            X_synthetic, y_synthetic = generate_synthetic_data(num_samples=num_generated_samples)
            yield X_synthetic, y_synthetic
    else:
        for X_real, y_real in real_dataloader:
            X_synthetic, y_synthetic = generate_synthetic_data(num_samples=num_generated_samples)
            X_batch = torch.cat([X_real, X_synthetic], dim=0).to(device)
            y_batch = torch.cat([y_real, y_synthetic], dim=0).to(device)
            yield X_batch, y_batch


net = DNN(input_dim=features_count, approach=approach, dropout_rate=dropout_rate).to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
if approach == 'pairwise':
    criterion = nn.CrossEntropyLoss()
elif approach == 'pointwise':
    criterion = nn.MSELoss()
early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=early_stopping_min_delta)

test_reader = DataLoaderTest(test_data_file, features_count=features_count, device=device)
test_reader_iter = iter(test_reader)

# Start with the pointwise approach because we need the number of data points to find the num_steps_per_epoch, then overwrite the train_reader_iter with the pairwise approach
X_train = np.load(os.path.join(raw_config['real_data_path'], 'X_num_train.npy'))
y_train = np.load(os.path.join(raw_config['real_data_path'], 'y_train.npy'))

# Create a dataloader for the training data using pytorch
train_reader_real = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float().to(device),
                                                   torch.from_numpy(y_train).float().to(device))
train_reader_iter_real = torch.utils.data.DataLoader(train_reader_real, batch_size=batch_size, shuffle=True)

num_steps_per_epoch = len(train_reader_iter_real)
wandb.config.update({"num_steps_per_epoch": num_steps_per_epoch})

if approach == 'pairwise':
    train_reader_real = DataLoaderTrain(train_data_file, approach=approach, features_count=features_count, device=device)
    train_reader_iter_real = iter(train_reader_real)


def train(net):
    net.train()
    train_loss = 0.0
    e_size = 0
    train_reader = mixed_dataloader(train_reader_iter_real, num_synthetic_data_in_batch, num_steps_per_epoch, device)
    train_reader_iter = iter(train_reader)
    for features, labels in train_reader_iter:
        e_size += 1
        if approach == 'pairwise':
            if e_size == num_steps_per_epoch:
                break
            out0 = net(features[0])
            out1 = net(features[1])
            out = torch.cat((out0, out1), dim=1)
            loss = criterion(out, labels)
        elif approach == 'pointwise':
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
        
        return avgp, avgndcg, val_loss


if __name__ == '__main__':
    print('Approach: {}'.format(approach))
    print(f"Values: P={p_value}, K={k_value}")
    print('Dataset: {}'.format(dataset))
    print('Number of learnable parameters: {}'.format(net.parameter_count()))

    best_val_loss = float('inf')
    best_model_state = None

    test(net, 0, 'n/a')
    for epoch in trange(num_epochs):
        train_loss = train(net)
        avgp, avgndcg, val_loss = test(net, epoch + 1, str(train_loss))

        wandb.log({'train_loss': train_loss, 'avgndcg': avgndcg, 'avgp': avgp, 'val_loss': val_loss})
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = net.state_dict().copy()
            
        # Early stopping check
        if early_stopping(val_loss):
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break


    # Load best model before saving
    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    # save model
    torch.save(net.state_dict(), model_output_file_path)
    print('Model saved to {}'.format(model_output_file_path))
    wandb.finish()
