import os
import numpy as np
import json
import shutil
from utils import set_all_seeds

# Set random seeds for reproducibility
seed = 42
set_all_seeds(seed)


def create_subsamples(k_values, idx_train, X_num_train, y_train, base_dir, info):
    for k in k_values:
        # Calculate the number of samples to retain
        n_samples = int(len(idx_train) * k)
        
        # Subsample the data
        subsample_idx = np.random.choice(len(idx_train), n_samples, replace=False)
        mask = np.zeros(len(idx_train), dtype=bool)
        mask[subsample_idx] = True
        
        # Extract sampled and non-sampled data
        idx_train_k = idx_train[mask]
        X_num_train_k = X_num_train[mask]
        y_train_k = y_train[mask]
        
        # Non-sampled data (if save_non_sampled is True)
        idx_train_non_k = idx_train[~mask]
        X_num_train_non_k = X_num_train[~mask]
        y_train_non_k = y_train[~mask]

        # Create a new directory for the subsample
        k_dir = os.path.join('data', f"{dataset}_k{k}")
        os.makedirs(k_dir, exist_ok=True)

        # Update the train_size in info.json
        info_updated = info.copy()
        info_updated["train_size"] = n_samples

        # Save the subsampled train data
        np.save(os.path.join(k_dir, "idx_train.npy"), idx_train_k)
        np.save(os.path.join(k_dir, "X_num_train.npy"), X_num_train_k)
        np.save(os.path.join(k_dir, "y_train.npy"), y_train_k)
        
        # Save the non-sampled train data
        np.save(os.path.join(k_dir, "idx_train_non.npy"), idx_train_non_k)
        np.save(os.path.join(k_dir, "X_num_train_non.npy"), X_num_train_non_k)
        np.save(os.path.join(k_dir, "y_train_non.npy"), y_train_non_k)

        # Copy the test and validation files directly
        for filename in ["idx_test.npy", "X_num_test.npy", "y_test.npy", 
                         "idx_val.npy", "X_num_val.npy", "y_val.npy"]:
            shutil.copy(os.path.join(base_dir, filename), os.path.join(k_dir, filename))

        # Save the updated info.json
        with open(os.path.join(k_dir, "info.json"), "w") as f:
            json.dump(info_updated, f)
            
        print(f"Created {n_samples} samples for k={k} in {k_dir}")
        print(f"Saved {len(idx_train_non_k)} non-sampled elements")



if __name__ == '__main__':
        # Base directory and k values
    dataset = "MSLR-Web30K"
    base_dir = os.path.join("..", "data", dataset, "npy", "Fold1")
    k_values = [1.0, 0.5, 0.25, 0.125, 0.0625]

    # Load train data and other files
    idx_train = np.load(os.path.join(base_dir, "idx_train.npy"))
    X_num_train = np.load(os.path.join(base_dir, "X_num_train.npy"))
    y_train = np.load(os.path.join(base_dir, "y_train.npy"))

    with open(os.path.join(base_dir, "info.json"), "r") as f:
        info = json.load(f)
    
    create_subsamples(k_values, idx_train, X_num_train, y_train, base_dir, info)