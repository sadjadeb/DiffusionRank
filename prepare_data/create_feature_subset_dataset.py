import os
import numpy as np
import json
import shutil
from utils import set_all_seeds

# Set random seeds for reproducibility
seed = 42
set_all_seeds(seed)

# Base directory and k values
dataset = "MQ2008"
num_subfeatures = 5
base_dir = os.path.join("data", dataset, "npy", "Fold1")

# Load train data and other files
idx_train = np.load(os.path.join(base_dir, "idx_train.npy"))
X_num_train = np.load(os.path.join(base_dir, "X_num_train.npy"))
y_train = np.load(os.path.join(base_dir, "y_train.npy"))

# Load test and validation data
idx_test = np.load(os.path.join(base_dir, "idx_test.npy"))
X_num_test = np.load(os.path.join(base_dir, "X_num_test.npy"))
y_test = np.load(os.path.join(base_dir, "y_test.npy"))

idx_val = np.load(os.path.join(base_dir, "idx_val.npy"))
X_num_val = np.load(os.path.join(base_dir, "X_num_val.npy"))
y_val = np.load(os.path.join(base_dir, "y_val.npy"))
    
if num_subfeatures > X_num_train.shape[1]:
    raise ValueError("Number of subfeatures cannot exceed the number of features")

# Randomly select `num_subfeatures` features
num_features = X_num_train.shape[1]
selected_features = np.random.choice(num_features, num_subfeatures, replace=False)
print(f"Selected features: {selected_features}")

# Create sub-datasets by selecting only the chosen features
X_num_train_sub = X_num_train[:, selected_features]
X_num_test_sub = X_num_test[:, selected_features]
X_num_val_sub = X_num_val[:, selected_features]

# Save the new datasets to files
output_dir = os.path.join("data", dataset, f"sub-{num_subfeatures}")
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, "idx_train.npy"), idx_train)
np.save(os.path.join(output_dir, "X_num_train.npy"), X_num_train_sub)
np.save(os.path.join(output_dir, "y_train.npy"), y_train)

np.save(os.path.join(output_dir, "idx_test.npy"), idx_test)
np.save(os.path.join(output_dir, "X_num_test.npy"), X_num_test_sub)
np.save(os.path.join(output_dir, "y_test.npy"), y_test)

np.save(os.path.join(output_dir, "idx_val.npy"), idx_val)
np.save(os.path.join(output_dir, "X_num_val.npy"), X_num_val_sub)
np.save(os.path.join(output_dir, "y_val.npy"), y_val)

# Save selected features for reference
selected_features_path = os.path.join(output_dir, "selected_features.json")
with open(selected_features_path, "w") as f:
    json.dump(selected_features.tolist(), f)

with open(os.path.join(base_dir, "info.json"), "r") as f:
    info = json.load(f)

# Update the number of numerical features
info["n_num_features"] = num_subfeatures

# Save the updated info.json to the output directory
with open(os.path.join(output_dir, "info.json"), "w") as f:
    json.dump(info, f, indent=4)

print(f"Sub-feature datasets created and saved in: {output_dir}")

