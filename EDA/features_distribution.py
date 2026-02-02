import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import set_all_seeds
from tqdm import trange

# Set the random seed for reproducibility
set_all_seeds()

dataset_name = 'MQ2007'

# Load the data
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_root, 'data', dataset_name, 'by_fraction', 'Fold1', f'k{k}')


X_train = np.load(os.path.join(data_dir, 'X_num_train.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train.npy'))

X_val = np.load(os.path.join(data_dir, 'X_num_val.npy'))
y_val = np.load(os.path.join(data_dir, 'y_val.npy'))

X_test = np.load(os.path.join(data_dir, 'X_num_test.npy'))
y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

# Concat features and labels of train, val, and test sets
X_train = np.concatenate((y_train.reshape(-1, 1), X_train), axis=1)
X_val = np.concatenate((y_val.reshape(-1, 1), X_val), axis=1)
X_test = np.concatenate((y_test.reshape(-1, 1), X_test), axis=1)


for y_index in trange(X_train.shape[1]):
    # plt.style.use('default')  # Using default style instead of seaborn
    plt.figure(figsize=(10, 6))

    # Create the density plots
    sns.kdeplot(data=X_train[:, y_index], label='Train Set', fill=True, alpha=0.5)
    sns.kdeplot(data=X_val[:, y_index], label='Validation Set', fill=True, alpha=0.5)
    sns.kdeplot(data=X_test[:, y_index], label='Test Set', fill=True, alpha=0.5)

    # Remove top and right spines
    sns.despine()

    plt.xlabel(f"Density of Data Points for index {y_index}", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Density Plot", fontsize=14)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join('output', 'distribution', f'density_plot.{dataset_name}.{y_index}.png'))
    
    plt.clf()
