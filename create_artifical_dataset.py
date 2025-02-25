import numpy as np
import os
import json
from utils import set_all_seeds

# Set random seeds for reproducibility
seed = 42
set_all_seeds(seed)

# Define weights such that w1 + w2 + w3 + w4 = 1
weights = np.array([0.3, 0.1, 0.2, 0.4])

# Define dataset sizes
train_size = 9630
val_size = 2707
test_size = 2874
total_size = train_size + val_size + test_size
n_features = 46

# Generate random weights for the features such that their sum equals 1
random_weights = np.random.rand(n_features)
normalized_weights = random_weights / np.sum(random_weights)

# Randomly sample features f1, f2, f3, f4
features = np.random.uniform(0, 1, size=(total_size, n_features))

# Compute the label y
labels = np.dot(features, normalized_weights)

# Create an index array
indices = np.arange(total_size)  # Incremental index from 0 to total_size - 1

# Split the dataset
train_features = features[:train_size]
train_labels = labels[:train_size]
train_indices = indices[:train_size]

val_features = features[train_size:train_size + val_size]
val_labels = labels[train_size:train_size + val_size]
val_indices = indices[train_size:train_size + val_size]

test_features = features[train_size + val_size:]
test_labels = labels[train_size + val_size:]
test_indices = indices[train_size + val_size:]

# Save features and labels for each split as separate NumPy arrays
data_path = os.path.join("data", f"artificial-{n_features}")
os.makedirs(data_path, exist_ok=True)

np.save(os.path.join(data_path, "X_num_train.npy"), train_features)
np.save(os.path.join(data_path, "y_train.npy"), train_labels)
np.save(os.path.join(data_path, "idx_train.npy"), train_indices)

np.save(os.path.join(data_path, "X_num_val.npy"), val_features)
np.save(os.path.join(data_path, "y_val.npy"), val_labels)
np.save(os.path.join(data_path, "idx_val.npy"), val_indices)

np.save(os.path.join(data_path, "X_num_test.npy"), test_features)
np.save(os.path.join(data_path, "y_test.npy"), test_labels)
np.save(os.path.join(data_path, "idx_test.npy"), test_indices)

dataset_info = {
    "name": f"artificial-{n_features}",
    "id": f"artificial{n_features}--default",
    "task_type": "regression",
    "n_num_features": n_features,
    "n_cat_features": 0,
    "train_size": train_size,
    "val_size": val_size,
    "test_size": test_size,
}

with open(os.path.join(data_path, "info.json"), "w") as f:
    json.dump(dataset_info, f)

print("Dataset created and saved successfully!")
