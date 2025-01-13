import os
import math
import numpy as np
import torch
import lightgbm as lgb
from tqdm import trange
import wandb
from argparse import ArgumentParser

import generative.lib as lib
from generative.tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from generative.tab_ddpm.modules import MLPDiffusion
from generative.scripts.utils_train import make_dataset
from utils import set_all_seeds, calculate_metrics

seed = 42
set_all_seeds(seed)

parser = ArgumentParser()
parser.add_argument('--p', type=float)
parser.add_argument('--k', type=float)
parser.add_argument('--sub', type=int, choices=[1, 2], help="When sub=2, it uses the generative model that partially trained with unlabeled data.")
args = parser.parse_args()

p_value = args.p
k_value = args.k
sub_exp_no = args.sub

approach = 'pointwise'  # Only pointwise supported for LightGBM
dataset = 'MQ2007'  # ['MQ2007', 'MQ2008', 'MSLR-Web10K', 'MSLR-Web30K']

# Set hyperparameters
device = torch.device("cuda:1")
features_count = 136 if 'MSLR' in dataset else 46

# LightGBM specific parameters
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'verbose': -1
}

# Set data paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_output_file_path = os.path.join('experiments', 'traditional', f"sub{sub_exp_no}", f"ltr.{dataset}.p{p_value}.k{k_value}.txt")

if sub_exp_no == 1:
    generative_experiment_id = f"exp_k{k_value}"
elif sub_exp_no == 2:
    generative_experiment_id = f"exp_k{k_value}_unlabeled"
generative_config_path = os.path.join('..', 'generative', 'experiments', 'MQ2007', generative_experiment_id, 'config.toml')
raw_config = lib.load_config(generative_config_path)

    
X_train = np.load(os.path.join(raw_config['real_data_path'], 'X_num_train.npy'))
y_train = np.load(os.path.join(raw_config['real_data_path'], 'y_train.npy'))

X_test = np.load(os.path.join(raw_config['real_data_path'], 'X_num_test.npy'))
y_test = np.load(os.path.join(raw_config['real_data_path'], 'y_test.npy'))
test_qids = np.load(os.path.join(raw_config['real_data_path'], 'idx_test.npy'))

data_len = len(X_train)
num_synthetic_data = int(data_len * p_value)
num_real_data = data_len - num_synthetic_data
print(f"Using {num_synthetic_data} synthetic data points and {num_real_data} real data points in whole batch with size {data_len}")

wandb.init(project=f"ltr_by_synthetic_{dataset}_sub{sub_exp_no}_{approach}_traditional", name=f"exp_p{p_value}_k{k_value}", group=f"{k_value}")
wandb.config.update({
    "p": p_value,
    "k": k_value,
    "data_length": data_len,
    "num_synthetic_data": num_synthetic_data,
    "num_real_data": num_real_data,
    "generative_config_path": generative_config_path,
    "lgb_params": lgb_params
})

# Load and setup diffusion model (same as original)
T_dict = raw_config['train']['T']
T = lib.Transformations(**T_dict)
D = make_dataset(
    os.path.join(raw_config['real_data_path']),
    T,
    num_classes=raw_config['model_params']['num_classes'],
    is_y_cond=raw_config['model_params']['is_y_cond'],
)
_, empirical_class_dist = torch.unique(torch.from_numpy(D.y['train']), return_counts=True)

K = np.array(D.get_category_sizes('train'))
if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
    K = np.array([0])

num_numerical_features_ = D.X_num['train'].shape[1] if D.X_num is not None else 0
d_in = np.sum(K) + num_numerical_features_
raw_config['model_params']['d_in'] = int(d_in)

model = MLPDiffusion(**raw_config['model_params'])
model_path = os.path.join('..', 'generative', raw_config['parent_dir'], 'model.pt')
model.load_state_dict(torch.load(model_path, map_location="cpu"))

diffusion = GaussianMultinomialDiffusion(
    **raw_config['diffusion_params'],
    num_classes=K,
    num_numerical_features=num_numerical_features_,
    denoise_fn=model,
    device=device,
)

diffusion.to(device)
diffusion.eval()


def generate_synthetic_data(batch_size=256, num_samples=0, num_numerical_features=features_count + 1):
    x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False, verbose=False)
    X_gen, y_gen = x_gen.numpy(), y_gen.numpy()

    num_numerical_features = num_numerical_features + 1
    X_num = D.num_transform.inverse_transform(X_gen[:, :num_numerical_features])
    X_num = X_num[:, :num_numerical_features]

    y_gen = X_num[:, 0]
    X_num = X_num[:, 1:]

    return X_num, y_gen


def test(model, X_test, y_test, qids):
    predictions = model.predict(X_test)
    results = {}
    
    # Group predictions by query
    for i, qid in enumerate(qids):
        if qid not in results:
            results[qid] = []
        results[qid].append((y_test[i], predictions[i]))
    
    avgndcg, avgp = calculate_metrics(results)
    
    return avgp, avgndcg

if __name__ == '__main__':
    print('Dataset:', dataset)
    print(f"Values: P={p_value}, K={k_value}")
    
    # Generate synthetic data if needed
    if num_synthetic_data > 0:
        X_synthetic, y_synthetic = generate_synthetic_data(num_samples=num_synthetic_data)
        
        X_train = X_train[:num_real_data]
        y_train = y_train[:num_real_data]
        
        # Combine real and synthetic data
        X_train = np.concatenate([X_train, X_synthetic], axis=0)
        y_train = np.concatenate([y_train, y_synthetic])
        
    print('Training data shape:', X_train.shape, y_train.shape)
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Train model
    print("Training LightGBM model...")
    model = lgb.train(lgb_params,
                     train_data,
                     callbacks=[lgb.log_evaluation(-1)])
    
    # Evaluate model
    print("Evaluating model...")
    avgp, avgndcg = test(model, X_test, y_test, test_qids)
    print(f"Results - P@10: {avgp:.4f}, NDCG@10: {avgndcg:.4f}")
    
    # Log metrics to wandb
    wandb.log({'avgndcg': avgndcg, 'avgp': avgp})
    
    # Save model
    model.save_model(model_output_file_path)
    print('Model saved to', model_output_file_path)
    wandb.finish()