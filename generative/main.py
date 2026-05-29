import glob
import json
import os
import pickle
import numpy as np
from tabdiff.modules.main_modules import UniModOnlyMLP
from tabdiff.modules.main_modules import Model
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from tabdiff.trainer import Trainer
import torch
import argparse
import warnings
import wandb
from sklearn.preprocessing import QuantileTransformer
from utils import set_all_seeds, load_config

warnings.filterwarnings('ignore')

# Set the random seed for reproducibility
seed = 42
set_all_seeds(seed)

def main(args):
    device = args.device

    ## Disable scientific numerical format
    np.set_printoptions(suppress=True)
    torch.set_printoptions(sci_mode=False)

    ## Get data info
    dataset = args.dataname
    k = args.k if args.k else 1.0
    if dataset == "MSLR-WEB10K" or dataset == "MSLR-WEB30K":
        d_numerical = 136
        threshold_of_neg = 1
    elif dataset == "MQ2007" or dataset == "MQ2008":
        d_numerical = 46
        threshold_of_neg = 0
    elif dataset == "Istella-S":
        d_numerical = 220
        threshold_of_neg = 1
    
    if args.approach == 'pointwise':
        categories = np.array([2])
    elif args.approach in ['pairwise', 'listwise_lambdarank']:
        if dataset == "MSLR-WEB10K" or dataset == "MSLR-WEB30K":
            categories = np.array([5])
        elif dataset == "MQ2007" or dataset == "MQ2008":
            categories = np.array([3])
        elif dataset == "Istella-S":
            categories = np.array([5])
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data', dataset, 'by_fraction', 'Fold1', f'k{k}')

    ## Set experiment name
    exp_name = args.exp_name
    if args.exp_name is None:
        exp_name = 'non_learnable_schedule' if args.non_learnable_schedule else 'learnable_schedule'
    
    ## Load configs
    curr_dir = 'tabdiff'
    config_path = f'{curr_dir}/tabdiff_configs.toml'
    raw_config = load_config(config_path)
    
    print(f"{args.mode.capitalize()} Mode is Enabled")
    num_samples_to_generate = None
    ckpt_path = args.ckpt_path
    
    exp_name += f"_{args.approach}_k{k}"
    
    if args.mode == 'train':
        print("NEW training is started")                
        if args.finetune:
            print("Finetuning is enabled, will load the finetune_ckpt_path")
            
            ckpt_path = args.finetune_ckpt_path
            raw_config['train']['main']['c_lambda'] = 1.0
            raw_config['train']['main']['d_lambda'] = 1.0
            if args.finetune_ckpt_path is None:
                raise ValueError("Please provide the finetune_ckpt_path to finetune the model!")
            if args.k is None:
                raise ValueError("Please provide the amount of training data to be used for finetuning, e.g., 0.25 means 25% of the training data will be used for finetuning")
    elif args.mode == 'test':
        args.no_wandb = True
        num_samples_to_generate = args.num_samples_to_generate
        ckpt_path = args.ckpt_path
        if ckpt_path is None:
            ckpt_parent_path = f"checkpoints/{dataset}/{exp_name}"
            ckpt_path_arr = glob.glob(f"{ckpt_parent_path}/best_model*")
            assert ckpt_path_arr, f"Cannot not infer ckpt_path from {ckpt_parent_path}, please make sure that you first train a model before testing!"
            ckpt_path = ckpt_path_arr[0]
        config_path = os.path.join(os.path.dirname(ckpt_path), 'config.pkl')
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                cached_raw_config = pickle.load(f)
                print(f"Found cached config at {config_path}")
        raw_config = cached_raw_config
    

    ## Creat model_save and result paths
    model_save_path =  f'checkpoints/{dataset}/{exp_name}'
    raw_config['model_save_path'] = model_save_path
    os.makedirs(model_save_path, exist_ok=True)
    
    ## Load training data
    raw_config['train']['main']['steps'] = args.steps
    raw_config['train']['main']['lr'] = args.lr
    raw_config['train']['main']['closs_weight_schedule'] = args.closs_weight_schedule
    raw_config['unimodmlp_params']['dim_t'] = args.dim_t
    raw_config['unimodmlp_params']['num_layers'] = args.num_layers
    raw_config['train']['main']['batch_size'] = args.batch_size
        
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    idx_train = np.load(os.path.join(data_dir, 'idx_train.npy'))

    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    idx_val = np.load(os.path.join(data_dir, 'idx_val.npy'))

    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    idx_test = np.load(os.path.join(data_dir, 'idx_test.npy'))

    if args.approach == 'pointwise':
        # Binarize labels
        y_train[y_train <= threshold_of_neg], y_train[y_train > threshold_of_neg] = 0, 1


    # Apply QuantileTransformer
    normalizer = QuantileTransformer(
                output_distribution='normal',
                n_quantiles=max(min(X_train.shape[0] // 30, 1000), 10),
                subsample=int(1e9),
                random_state=seed,
            )
    X_train = normalizer.fit_transform(X_train)
    X_val = normalizer.transform(X_val)
    X_test = normalizer.transform(X_test)
    
    # concat the X_train and y_train
    X_train = np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1)
    X_val = np.concatenate([X_val, y_val.reshape(-1, 1)], axis=1)
    X_test = np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1)
    
    # For pairwise/listwise training, organize data by query ID
    train_data_by_qid = None
    if args.approach in ['pairwise', 'listwise_lambdarank']:
        train_data_by_qid = {}
        for i in range(len(X_train)):
            qid = idx_train[i]
            if qid not in train_data_by_qid:
                train_data_by_qid[qid] = {'features': [], 'labels': [], 'indices': []}
            train_data_by_qid[qid]['features'].append(X_train[i])
            train_data_by_qid[qid]['labels'].append(y_train[i])
            train_data_by_qid[qid]['indices'].append(i)
        
        # Filter out queries with only one document (can't form pairs)
        num_queries_before = len(train_data_by_qid)
        train_data_by_qid = {qid: data for qid, data in train_data_by_qid.items() 
                             if len(data['labels']) > 1}
        num_queries_after = len(train_data_by_qid)
        num_filtered = num_queries_before - num_queries_after
        print(f"Organized training data into {num_queries_after} queries with multiple documents (filtered out {num_filtered} queries with only one document)")

    # Create PyTorch tensors from numpy arrays
    train_data = torch.from_numpy(X_train).float().to(device)
    val_data = torch.from_numpy(X_val).float().to(device)
    test_data = torch.from_numpy(X_test).float().to(device)
    
    ## Load the module and models
    raw_config['unimodmlp_params']['approach'] = args.approach
    raw_config['unimodmlp_params']['d_numerical'] = d_numerical
    raw_config['unimodmlp_params']['categories'] = (categories).tolist()  # add one for the mask category
            
    backbone = UniModOnlyMLP(**raw_config['unimodmlp_params'])
    model = Model(backbone, **raw_config['diffusion_params']['edm_params'])
    model.to(device)
    
    if not args.non_learnable_schedule:
        raw_config['diffusion_params']['scheduler'] = 'power_mean_per_column'
        raw_config['diffusion_params']['cat_scheduler'] = 'log_linear_per_column'
    
    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories,
        num_numerical_features=d_numerical,
        denoise_fn=model,
        **raw_config['diffusion_params'],
        device=device,
    )
    num_params = sum(p.numel() for p in diffusion.parameters())
    print("The number of parameters = ", num_params)
    diffusion.to(device)
    diffusion.train()

    ## Print the configs
    printed_configs = json.dumps(raw_config, default=lambda x: int(x) if isinstance(x, np.int64) else x, indent=4)
    print(f"The config of the current run is : \n {printed_configs}")
    
    ## Enable Wandb
    project_name = f"DiffusionRank_{dataset}"
    raw_config['project_name'] = project_name
    logger = wandb.init(
        project=raw_config['project_name'], 
        name=exp_name,
        config=raw_config,
        mode='disabled' if args.debug or args.no_wandb else 'online',
    )

    ## Load Trainer
    sample_batch_size = raw_config['sample']['batch_size']
    trainer = Trainer(
        diffusion,
        train_data,
        val_data,
        idx_val,
        test_data,
        idx_test,
        d_numerical,
        categories,
        logger,
        **raw_config['train']['main'],
        sample_batch_size=sample_batch_size,
        num_samples_to_generate=num_samples_to_generate,
        model_save_path=raw_config['model_save_path'],
        device=device,
        ckpt_path=ckpt_path,
        is_finetune=args.finetune,
        raw_data_dir=data_dir,
        bell_mu=args.bell_mu,
        bell_peak=args.bell_peak,
        bell_sigma=args.bell_sigma,
        approach=args.approach,
        train_data_by_qid=train_data_by_qid
    )
    if args.mode == 'test':
        predictions_save_path = os.path.join('predictions', f'ltr.{dataset}.{args.approach}.k{k}.{exp_name}.best.txt')
        
        predictions = trainer.predict(test_data, idx_test)
        
        with open(predictions_save_path, 'w') as f:
            f.write('qid true_label pred_label\n')
            for qid, values in predictions.items():
                for true_label, pred_label in values:
                    f.write(f'{qid} {true_label} {pred_label}\n')
        print('Results saved to {}'.format(predictions_save_path))
    else:
        ## Save config
        config_save_path = raw_config['model_save_path']
        with open (os.path.join(config_save_path, 'config.pkl'), 'wb') as f:
            pickle.dump(raw_config, f)
        trainer.run_loop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of TabDiff')

    # General configs
    parser.add_argument('--dataname', type=str, default='adult', help='Name dataset, one of those in data/ dir')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--method', type=str, default='tabdiff', help='Currently we only release our model TabDiff. Baselines will be released soon.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no_wandb', action='store_true', help='disable wandb')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name, used to name log directories and the wandb run name')
    parser.add_argument('--approach', type=str, default='pointwise', choices=['pointwise', 'pairwise', 'listwise_lambdarank'], help='Training approach')
    
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for training and evaluation')
    parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--closs_weight_schedule', type=str, default='anneal')
    parser.add_argument('--dim_t' , type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--steps', type=int, default=15000, help='Number of training steps')
    
    parser.add_argument('--bell_mu', type=float, default=None, help='mu parameter for the bell noise schedule')
    parser.add_argument('--bell_peak', type=float, default=None, help='peak parameter for the bell noise schedule')
    parser.add_argument('--bell_sigma', type=float, default=None, help='sigma parameter for the bell noise schedule. If None, it will be set to 1/(2*mu)')

    # Configs for tabdiff
    parser.add_argument('--non_learnable_schedule', action='store_true', help='disable learnable noise schedule')
    
    # Configs for testing tabdiff
    parser.add_argument('--num_samples_to_generate', type=int, default=None, help='Number of samples to be generated while testing')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to the model checkpoint to be tested')
    parser.add_argument('--report', action='store_true', help="Report testing mode: this mode sequentially runs <num_runs> test runs and report the avg and std")
    parser.add_argument('--num_runs', type=int, default=20, help="Number of runs to be averaged in the report testing mode")
    
    # Configs for imputation
    parser.add_argument('--impute', action='store_true')
    parser.add_argument('--impute_condition', type=str, default="x_t")

    # Configs for fractional training and finetuning
    parser.add_argument('--k', type=float, help='Portion of the training data to be used for training, e.g., 0.25 means 25%% of the training data will be used')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--finetune_ckpt_path', type=str, default=None, help='Path to the model checkpoint to be finetuned')

    args = parser.parse_args()
    
    main(args)