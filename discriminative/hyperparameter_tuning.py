import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
import datetime

from model import DNN
from data_loader import DataLoaderTrain, DataLoaderTest
from utils import set_all_seeds, calculate_metrics

seed = 42
set_all_seeds(seed)

approach = 'pointwise' # ['pointwise', 'pairwise']
dataset = 'MQ2007' # ['MQ2007', 'MQ2008', 'MSLR-Web10K', 'MSLR-Web30K']

# Set hyperparameters
device = torch.device("cuda:3")
features_count = 136 if 'MSLR' in dataset else 46
num_steps_per_epoch = 2048
num_epochs = 10

# Set data paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_root, 'data', dataset, 'raw', 'Fold1')
train_data_file = os.path.join(data_dir, 'train.txt')
test_data_file = os.path.join(data_dir, 'test.txt')

# Hyperparameter grid
param_grid = {
    'num_hidden_nodes': [64, 128, 192],
    'num_hidden_layers': [2, 3, 4],
    'learning_rate': [0.0001, 0.0002, 0.0005],
    'dropout_rate': [0.0, 0.1, 0.2, 0.3]
}

output_file = os.path.join("output", f"hyperparameter_tuning_{dataset}_{approach}_results.txt")


def write_to_file(message, file_path=output_file, mode='a'):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Format the timestamp
    with open(file_path, mode) as f:
        f.write(f'{timestamp} - {message}\n')


# Load train and test data
train_reader = DataLoaderTrain(train_data_file, approach=approach, features_count=features_count, device=device)
train_reader_iter = iter(train_reader)
test_reader = DataLoaderTest(test_data_file, features_count=features_count, device=device)
test_reader_iter = iter(test_reader)


def train_and_test(net, optimizer, criterion):
    epoch_results = []
    for epoch in range(num_epochs):
        train_loss = 0.0
        net.train()
        for _ in range(num_steps_per_epoch):
            features, labels = next(train_reader_iter)
            if approach == 'pairwise':
                out0 = net(features[0])
                out1 = net(features[1])
                out = torch.cat((out0, out1), dim=1)
                loss = criterion(out, labels)
            elif approach == 'pointwise':
                labels = labels.view(-1, 1).float()
                out = net(features)
                loss = criterion(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / num_steps_per_epoch
        avgp, avgndcg = test(net)
        performance = (avgp + avgndcg) / 2

        epoch_result = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'avgp': avgp,
            'avgndcg': avgndcg,
            'performance': performance
        }
        epoch_results.append(epoch_result)

        message = f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_train_loss:.4f}, AvgP: {avgp:.4f}, AvgNDCG: {avgndcg:.4f}, Performance: {performance:.4f}"
        print(message)
        write_to_file(message)

    return epoch_results


def test(net):
    net.eval()

    with torch.no_grad():
        results = {}
        for features, qids, labels, cnt in test_reader_iter:
            out = net(features).data.cpu()
            row_cnt = len(qids)
            for i in range(row_cnt):
                if qids[i] not in results:
                    results[qids[i]] = []
                results[qids[i]].append((labels[i], out[i][0]))

    avgndcg, avgp = calculate_metrics(results)

    return avgp, avgndcg


def hyperparameter_tuning():
    best_performance = 0
    best_params = {}
    results = []

    total_combinations = np.prod([len(v) for v in param_grid.values()])
    write_to_file(f"Total hyperparameter combinations to try: {total_combinations}\n")

    for i, params in enumerate(product(*param_grid.values()), 1):
        current_params = dict(zip(param_grid.keys(), params))
        message = f"\nTrying parameters ({i}/{total_combinations}): {current_params}"
        print(message)
        write_to_file(message)

        net = DNN(input_dim=features_count,
                  num_hidden_nodes=current_params['num_hidden_nodes'],
                  num_hidden_layers=current_params['num_hidden_layers'],
                  dropout_rate=current_params['dropout_rate'],
                  approach=approach
                  ).to(device)
        optimizer = optim.Adam(net.parameters(), lr=current_params['learning_rate'])
        if approach == 'pairwise':
            criterion = nn.CrossEntropyLoss()
        elif approach == 'pointwise':
            criterion = nn.MSELoss()

        epoch_results = train_and_test(net, optimizer, criterion)

        # Get the best performance for this hyperparameter combination
        best_epoch_result = max(epoch_results, key=lambda x: x['performance'])
        results.append((current_params, best_epoch_result))

        message = f"Best epoch: {best_epoch_result['epoch']}, Performance: {best_epoch_result['performance']:.4f} (AvgP: {best_epoch_result['avgp']:.4f}, AvgNDCG: {best_epoch_result['avgndcg']:.4f})"
        print(message)
        write_to_file(message)

        if best_epoch_result['performance'] > best_performance:
            best_performance = best_epoch_result['performance']
            best_params = current_params

    message = "\nHyperparameter tuning completed."
    print(message)
    write_to_file(message)

    message = f"Best parameters: {best_params}"
    print(message)
    write_to_file(message)

    message = f"Best performance: {best_performance:.4f}"
    print(message)
    write_to_file(message)

    # Sort and print all results
    sorted_results = sorted(results, key=lambda x: x[1]['performance'], reverse=True)
    message = "\nAll results (sorted by best performance):"
    print(message)
    write_to_file(message)

    for params, best_result in sorted_results:
        message = f"Params: {params}\n"
        message += f"Best Epoch: {best_result['epoch']}, Performance: {best_result['performance']:.4f}, AvgP: {best_result['avgp']:.4f}, AvgNDCG: {best_result['avgndcg']:.4f}, Train Loss: {best_result['train_loss']:.4f}\n"
        print(message)
        write_to_file(message)


if __name__ == "__main__":
    print('Dataset: {}'.format(dataset))
    print('Approach: {}'.format(approach))
    write_to_file(f'Dataset: {dataset}')
    write_to_file(f'Approach: {approach}')
    write_to_file(f'Number of epochs: {num_epochs}')
    hyperparameter_tuning()
    print(f"\nResults have been saved to {output_file}")
