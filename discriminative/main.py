import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import RankingDataset
from model import RankingModel
import os
from tqdm import trange


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in trange(num_epochs, desc='Epoch'):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.float(), labels.float().unsqueeze(1)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Validate the model
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.float(), labels.float().unsqueeze(1)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}")


def test_model(model, test_loader):
    criterion = nn.MSELoss()
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.float(), labels.float().unsqueeze(1)
            outputs = model(features)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * features.size(0)

    test_loss = test_loss / len(test_loader.dataset)
    print(f"Test Loss (RMSE): {np.sqrt(test_loss):.4f}")


# Define paths to the validation and test datasets
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_file_path = os.path.join(base_dir, 'data', 'Fold1', 'train.txt')
val_file_path = os.path.join(base_dir, 'data', 'Fold1', 'vali.txt')
test_file_path = os.path.join(base_dir, 'data', 'Fold1', 'test.txt')

# Initialize the datasets
train_dataset = RankingDataset(train_file_path)
val_dataset = RankingDataset(val_file_path)
test_dataset = RankingDataset(test_file_path)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"Datasets loaded")


# Get the input dimension from the dataset
input_dim = train_dataset.data.shape[1]

# Initialize the model
model = RankingModel(input_dim)
print(f"Model initialized")

# Train the model
train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001)

# # Test the model
# test_model(model, test_loader)
