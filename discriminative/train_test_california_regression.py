import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import set_all_seeds
from sklearn.preprocessing import StandardScaler
from tqdm import trange


seed = 42
set_all_seeds(seed)

dataset = 'california'

# Set hyperparameters
device = torch.device("cuda:1")
features_count = 8
num_steps_per_epoch = 2048
num_epochs = 1
dropout_rate = 0.2 if 'MSLR' in dataset else 0.1
learning_rate = 5e-4 if 'MSLR' in dataset else 1e-5
num_hidden_nodes = 256 if 'MSLR' in dataset else 128
batch_size = 1024

# Set data paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_root, 'data', dataset)

X_train = np.load(os.path.join(data_dir, 'X_num_train.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
X_test = np.load(os.path.join(data_dir, 'X_num_test.npy'))
y_test = np.load(os.path.join(data_dir, 'y_test.npy'))


# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)

# Initialize the model, loss function, and optimizer
input_dim = X_train.shape[1]
model = MLP(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, X_train, y_train, criterion, optimizer, epochs=50, batch_size=32):
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in trange(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")


# Evaluate the model
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        loss = criterion(predictions, y_test)
        mae = torch.mean(torch.abs(predictions - y_test))
    print(f"Test Loss: {loss.item():.4f}")
    print(f"Test MAE: {mae.item():.4f}")


# Train the model
train_model(model, X_train, y_train, criterion, optimizer)
# Evaluate on the test set
evaluate_model(model, X_test, y_test)

# Optional: Save the trained model
torch.save(model.state_dict(), 'mlp_california_housing_model.pth')

