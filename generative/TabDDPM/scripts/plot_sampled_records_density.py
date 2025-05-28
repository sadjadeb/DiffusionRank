import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

dataset_name = 'MQ2007'
# experiment_id = 'exp_k1.0'
experiment_id = 'exp_2025-03-18_03-51-36'

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
trained_model_path = os.path.join(project_root, 'experiments', dataset_name, experiment_id)
X_sampled = np.load(os.path.join(trained_model_path, 'X_num_sampled.npy'))
y_sampled = np.load(os.path.join(trained_model_path, 'y_sampled.npy'))

if dataset_name == 'california':
    X_real = np.load(f'/mnt/data/sajadeb/Gen_LTR/data/{dataset_name}/X_num_train.npy')
    y_real = np.load(f'/mnt/data/sajadeb/Gen_LTR/data/{dataset_name}/y_train.npy')
else:
    X_real = np.load(f'/mnt/data/sajadeb/Gen_LTR/data/{dataset_name}/npy/Fold1/X_num_train.npy')
    y_real = np.load(f'/mnt/data/sajadeb/Gen_LTR/data/{dataset_name}/npy/Fold1/y_train.npy')


# make both the same size
if X_real.shape[0] > X_sampled.shape[0]:
    X_real = X_real[:X_sampled.shape[0], :]
else:
    X_sampled = X_sampled[:X_real.shape[0], :]

print(f"X_real shape: {X_real.shape}")
print(f"X_sampled shape: {X_sampled.shape}")
print(f"y_real shape: {y_real.shape}")
print(f"y_sampled shape: {y_sampled.shape}")


plt.figure(figsize=(10, 6))
for index in [0, 2, 15, 31, 43]:
    print(f"Plotting density plot for index {index}...")
    # Create the density plots
    if index == 0:
        sns.kdeplot(data=y_real, label='Original Values', fill=True, alpha=0.5)
        sns.kdeplot(data=y_sampled, label='Sampled Values', fill=True, alpha=0.5)        
    else:
        index -= 1
        sns.kdeplot(data=X_real[:, index], label='Original Values', fill=True, alpha=0.5)
        sns.kdeplot(data=X_sampled[:, index], label='Sampled Values', fill=True, alpha=0.5)
    
    sns.despine()

    plt.xlabel(f"Density of Sampled Data Points - index {index}", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Density Plot", fontsize=14)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(trained_model_path, 'figs', f'density_plot.sampled.{index}.png'))
    
    plt.clf()
