import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import lib


def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)

def make_dataset(
    data_path: str,
    T: lib.Transformations,
    num_classes: int,
    is_y_cond: bool,
    has_splitted_data: bool = False
):
    variations = ['train', 'val', 'test']
    if has_splitted_data:
        variations.append('train_non')
        
    if num_classes > 0:
        # classification
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) or not is_y_cond else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} 

        for split in variations:
            X_num_t, X_cat_t, y_t = lib.read_pure_data(data_path, split)
            if not is_y_cond:
                X_cat_t = concat_y_to_X(X_cat_t, y_t)
            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                X_cat[split] = X_cat_t
            y[split] = y_t
    else:
        # regression
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) or not is_y_cond else None
        y = {}
        

        for split in variations:
            X_num_t, X_cat_t, y_t = lib.read_pure_data(data_path, split)
            if not is_y_cond:
                X_num_t = concat_y_to_X(X_num_t, y_t)
            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                X_cat[split] = X_cat_t
            y[split] = y_t

    info = lib.load_json(os.path.join(data_path, 'info.json'))

    D = lib.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=lib.TaskType(info['task_type']),
        n_classes=info.get('n_classes')
    )
    
    return lib.transform_dataset(D, T, None)


def plot_inpainting_outputs(parent_dir, y_index, D, X_unnorm, X_num, X_predicted_noisy, y_pred, strategy):
    fig_save_path = os.path.join(parent_dir, 'figs')
    os.makedirs(fig_save_path, exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    sns.histplot(D.X_num['train'][:, y_index], bins=20, kde=True, color='blue', alpha=0.7)
    plt.title(f'Distribution of index {y_index}', fontsize=16)
    plt.xlabel('Value', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(os.path.join(parent_dir, 'figs', f'normalized_distribution.{y_index}.png'))
    plt.clf()
    
    
    # plot pred labels
    sorted_y_pred = np.sort(y_pred)
    x = np.arange(len(sorted_y_pred))

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, sorted_y_pred, alpha=0.6, s=20)  # alpha controls transparency, s controls point size
    plt.title(f'Data Points Visualization for index {y_index}', fontsize=14)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.8)
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
    plt.minorticks_on()  # Enable minor ticks
    plt.savefig(os.path.join(parent_dir, 'figs', f'scatter_plot.{strategy}.{y_index}.png'))
    plt.clf()
    

    # Set the style and figure size
    X_predicted_noisy = X_predicted_noisy.numpy()

    plt.figure(figsize=(10, 6))
    # Create the density plots
    sns.kdeplot(data=X_unnorm[:, y_index], label='Original Values', fill=True, alpha=0.5)
    sns.kdeplot(data=X_predicted_noisy[:, y_index], label='Original Values + Noise', fill=True, alpha=0.5)
    sns.kdeplot(data=X_num[:, y_index], label='Predicted Values', fill=True, alpha=0.5)
    
    sns.despine()

    plt.xlabel(f"Density of Data Points for index {y_index}", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Density Plot", fontsize=14)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(parent_dir, 'figs', f'density_plot.{strategy}.{y_index}.png'))
    