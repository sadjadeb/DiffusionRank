import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

dataset_name = 'MQ2008'
experiment_id = 'exp_k1.0'
# experiment_id = 'exp_2025-01-23_18-28-12'

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
trained_model_path = os.path.join(project_root, 'experiments', dataset_name, experiment_id)
X_sampled = np.load(os.path.join(trained_model_path, 'X_num_sampled.npy'))

if dataset_name == 'california':
    X_real = np.load(f'/mnt/data/sajadeb/Gen_LTR/data/{dataset_name}/X_num_train.npy')
else:
    X_real = np.load(f'/mnt/data/sajadeb/Gen_LTR/data/{dataset_name}/npy/Fold1/X_num_train.npy')

plt.figure(figsize=(10, 6))

for y_index in [0, 2, 15, 31, 43]:
    print(f"Plotting density plot for index {y_index}...")
    # Create the density plots
    sns.kdeplot(data=X_real[:, y_index], label='Original Values', fill=True, alpha=0.5)
    sns.kdeplot(data=X_sampled[:, y_index], label='Sampled Values', fill=True, alpha=0.5)
    
    sns.despine()

    plt.xlabel(f"Density of Data Points - {dataset_name} - index {y_index}", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Density Plot", fontsize=14)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(trained_model_path, 'figs', f'density_plot.sampled.{dataset_name}.{y_index}.png'))
    
    plt.clf()
