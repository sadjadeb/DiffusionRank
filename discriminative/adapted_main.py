"""
This script has been adapted from the original script from the following repository:
https://github.com/spacemanidol/AFIRMDeepLearning2020/blob/master/LTR.ipynb 
"""


import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
from model import DNN
from data_loader import DataLoaderTrain, DataLoaderTest
from utils import set_all_seeds, calculate_metrics

# Set random seeds for reproducibility
seed = 42
set_all_seeds(seed)

approach = 'pointwise' # ['pointwise', 'pairwise']
dataset = 'MQ2007' # ['MQ2007', 'MQ2008', 'MSLR-Web10K', 'MSLR-Web30K']

# Set hyperparameters
device = torch.device("cuda:3")
features_count = 136 if 'MSLR' in dataset else 46
num_steps_per_epoch = 2048
num_epochs = 10
dropout_rate = 0.1
learning_rate = 1e-5

# Set data paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_root, 'data', dataset, 'raw', 'Fold1')
train_data_file = os.path.join(data_dir, 'train.txt')
test_data_file = os.path.join(data_dir, 'test.txt')
model_output_file_path = f"ltr.{dataset}.{num_epochs}.pth"

# Load train and test data
train_reader = DataLoaderTrain(train_data_file, approach=approach, features_count=features_count, device=device)
train_reader_iter = iter(train_reader)
test_reader = DataLoaderTest(test_data_file, features_count=features_count, device=device)
test_reader_iter = iter(test_reader)

net = DNN(input_dim=features_count, approach=approach, dropout_rate=dropout_rate).to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
if approach == 'pairwise':
    criterion = nn.CrossEntropyLoss()
elif approach == 'pointwise':
    criterion = nn.MSELoss()
    

def train(net):
    net.train()
    train_loss = 0.0
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
    
    return train_loss / num_steps_per_epoch


def test(net, epoch, train_loss):
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
        
        print(f'epoch:{epoch}, loss: {train_loss}, avgp: {avgp}, avgndcg: {avgndcg}')


if __name__ == '__main__':
    print('Approach: {}'.format(approach))
    print('Dataset: {}'.format(dataset))
    print('Number of learnable parameters: {}'.format(net.parameter_count()))

    test(net, 0, 'n/a')
    for epoch in range(num_epochs):
        train_loss = train(net)    
        test(net, epoch + 1, str(train_loss))

    # save model
    torch.save(net.state_dict(), model_output_file_path)
    print('Model saved to {}'.format(model_output_file_path))
