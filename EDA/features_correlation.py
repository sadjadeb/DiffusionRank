import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from utils import set_all_seeds

seed = 42
set_all_seeds(seed)

device = 'cuda:3'
dataset_name = 'MSLR-WEB10K'

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
train_data_path = os.path.join(project_root, 'data', dataset_name, 'npy', 'Fold1', 'X_num_train.npy')
train_label_path = os.path.join(project_root, 'data', dataset_name, 'npy', 'Fold1', 'y_train.npy')
test_data_path = os.path.join(project_root, 'data', dataset_name, 'npy', 'Fold1', 'X_num_test.npy')
test_label_path = os.path.join(project_root, 'data', dataset_name, 'npy', 'Fold1', 'y_test.npy')

X_train = np.load(train_data_path)
y_train = np.load(train_label_path)
X_test = np.load(test_data_path)
y_test = np.load(test_label_path)

X_train = np.concatenate((y_train.reshape(-1, 1), X_train), axis=1)
X_test = np.concatenate((y_test.reshape(-1, 1), X_test), axis=1)

num_features = X_train.shape[1]

# Define a simple feed-forward neural network
class FeedForwardNN(nn.Module):
    def __init__(self, input_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Loop over each feature to predict it using the rest
mse_scores = {}
for i in range(num_features):
    # Combine features and target into one array for normalization
    target_index = i
    train_combined = np.hstack((X_train, X_train[:, [target_index]]))
    test_combined = np.hstack((X_test, X_test[:, [target_index]]))

    # Normalize the combined data
    scaler = StandardScaler()
    train_combined = scaler.fit_transform(train_combined)
    test_combined = scaler.transform(test_combined)

    # Separate normalized features and target for training and testing
    X_train_features = np.delete(train_combined, -1, axis=1)
    y_train = train_combined[:, -1]

    X_test_features = np.delete(test_combined, -1, axis=1)
    y_test = test_combined[:, -1]

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_features, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_features, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    # Initialize the model, loss function, and optimizer
    model = FeedForwardNN(input_size=X_train_features.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        
        # Calculate MSE on the normalized values
        mse = criterion(y_pred, y_test_tensor).item()
        if i == 0:
            mse_scores['label'] = mse
            print(y_pred)
            print('==========================')
            print(y_test_tensor)
        else:
            mse_scores[f'{i}'] = mse
            

    print(f'MSE for predicting feature_{i}: {mse:.4f}')

# Visualize the MSEs in a bar plot
plt.figure(figsize=(10, 6))
plt.bar(mse_scores.keys(), mse_scores.values(), color='skyblue')
plt.xlabel('Features')
plt.ylabel('MSE')
plt.title(f'Mean Squared Error for Predicting Each Feature in {dataset_name} Dataset')
plt.xticks(rotation=45)
plt.savefig(f'output/mse_scores_{dataset_name.lower()}_normalized.png')
