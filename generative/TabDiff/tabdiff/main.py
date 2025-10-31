import glob
import json
import os
import pickle
import random
import numpy as np
from tabdiff.modules.main_modules import UniModMLP, UniModOnlyMLP
from tabdiff.modules.main_modules import Model
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from tabdiff.trainer import Trainer
import src
import torch
import argparse
import warnings
import wandb
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings('ignore')


def main(args):
    device = args.device

    ## Disable scientific numerical format
    np.set_printoptions(suppress=True)
    torch.set_printoptions(sci_mode=False)

    ## Get data info
    dataname = args.dataname
    data_dir = f'data/{dataname}'

    ## Set experiment name
    exp_name = args.exp_name
    if args.exp_name is None:
        exp_name = 'non_learnable_schedule' if args.non_learnable_schedule else 'learnable_schedule'
    
    ## Load configs
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = f'{curr_dir}/configs/tabdiff_configs.toml'
    raw_config = src.load_config(config_path)
    
    print(f"{args.mode.capitalize()} Mode is Enabled")
    num_samples_to_generate = None
    ckpt_path = None
    
    if args.k:
        print(f"Training with {args.k} portion of the training data")
        
        exp_name += f"_k{args.k}"
        data_dir = f'data/{dataname}_k{args.k}'
        if not os.path.exists(data_dir):
            raise ValueError(f"The data directory {data_dir} does not exist, please make sure that you first prepare the data for finetuning!")
    
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
        num_samples_to_generate = args.num_samples_to_generate
        ckpt_path = args.ckpt_path
        if ckpt_path is None:
            ckpt_parent_path = f"{curr_dir}/ckpt/{dataname}/{exp_name}"
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
    model_save_path, result_save_path = None, None
    if args.mode == 'train':
        model_save_path = 'debug/ckpt' if args.debug else f'{curr_dir}/ckpt/{dataname}/{exp_name}'
        result_save_path = model_save_path.replace('ckpt', 'result')  #i.e., f'{curr_dir}/results/{dataname}/{exp_name}'
    elif args.mode == 'test':
        if args.report:
            result_save_path = f"eval/report_runs/{exp_name}/{dataname}"
        else:
            result_save_path = os.path.dirname(ckpt_path).replace('ckpt', 'result')    # infer the exp_name from the ckpt_name
    raw_config['model_save_path'] = model_save_path
    raw_config['result_save_path'] = result_save_path
    if model_save_path is not None:
        os.makedirs(model_save_path, exist_ok=True)
    if result_save_path is not None:
        os.makedirs(result_save_path, exist_ok=True)
    
    ## Make everything determinstic if needed
    raw_config['deterministic'] = args.deterministic
    if args.deterministic:
        print("DETERMINISTIC MODE is enabled!!!")
        seed = 42
        ## Set global random seeds
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        ## Ensure deterministic CUDA operations
        os.environ['PYTHONHASHSEED'] = '42'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # or ':16:8'
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    ## Load training data
    raw_config['train']['main']['steps'] = args.steps
    raw_config['train']['main']['lr'] = args.lr
    raw_config['train']['main']['closs_weight_schedule'] = args.closs_weight_schedule
    raw_config['unimodmlp_params']['dim_t'] = args.dim_t
    raw_config['unimodmlp_params']['num_layers'] = args.num_layers
    raw_config['train']['main']['batch_size'] = args.batch_size
    
    k = args.k if args.k else 1.0
    d_numerical = 136 if 'MSLR' in dataname else 46
    categories = np.array([2])
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    
    X_train = np.load(os.path.join(project_root, 'data', dataname, 'by_fraction', 'Fold1', f'k{k}', 'X_num_train.npy'))
    y_train = np.load(os.path.join(project_root, 'data', dataname, 'by_fraction', 'Fold1', f'k{k}', 'y_train.npy'))
    X_train_unlabeled = np.load(os.path.join(project_root, 'data', dataname, 'by_fraction', 'Fold1', f'k{k}', 'X_num_train_non.npy'))
    # Replace all labels greater than 1 with 1
    y_train[y_train > 1] = 1

    X_val = np.load(os.path.join(project_root, 'data', dataname, 'by_fraction', 'Fold1', f'k{k}', 'X_num_val.npy'))
    y_val = np.load(os.path.join(project_root, 'data', dataname, 'by_fraction', 'Fold1', f'k{k}', 'y_val.npy'))
    idx_val = np.load(os.path.join(project_root, 'data', dataname, 'by_fraction', 'Fold1', f'k{k}', 'idx_val.npy'))
    # Replace all labels greater than 1 with 1
    y_val[y_val > 1] = 1

    X_test = np.load(os.path.join(project_root, 'data', dataname, 'by_fraction', 'Fold1', f'k{k}', 'X_num_test.npy'))
    y_test = np.load(os.path.join(project_root, 'data', dataname, 'by_fraction', 'Fold1', f'k{k}', 'y_test.npy'))
    idx_test = np.load(os.path.join(project_root, 'data', dataname, 'by_fraction', 'Fold1', f'k{k}', 'idx_test.npy'))
    # Replace all labels greater than 1 with 1
    y_test[y_test > 1] = 1

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
    print(f"Train data shape: {X_train.shape}, Val data shape: {X_val.shape}, Test data shape: {X_test.shape}")
    
    # Create a dataloader for the training data using pytorch
    train_data = torch.from_numpy(X_train).float().to(device)
    # Create a dataloader for the validation data using pytorch
    val_data = torch.from_numpy(X_val).float().to(device)
    # Create a dataloader for the test data using pytorch
    test_data = torch.from_numpy(X_test).float().to(device)
    
    
    ## Load the module and models
    raw_config['unimodmlp_params']['d_numerical'] = d_numerical
    raw_config['unimodmlp_params']['categories'] = (categories+1).tolist()  # add one for the mask category
            
    backbone = UniModOnlyMLP(**raw_config['unimodmlp_params'])
    model = Model(backbone, **raw_config['diffusion_params']['edm_params'])
    model.to(device)
    
    if args.impute:
        raw_config['diffusion_params']['num_timesteps'] = 1

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
    project_name = f"tabdiff_{dataname}"
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
        k,
        logger,
        **raw_config['train']['main'],
        sample_batch_size=sample_batch_size,
        num_samples_to_generate=num_samples_to_generate,
        model_save_path=raw_config['model_save_path'],
        result_save_path=raw_config['result_save_path'],
        device=device,
        ckpt_path=ckpt_path,
        is_finetune=args.finetune,
        raw_data_dir=data_dir,
        bell_mu=args.bell_mu,
        bell_peak=args.bell_peak,
        bell_sigma=args.bell_sigma
    )
    if args.mode == 'test':
        if args.impute:
            imputed_sample_save_dir = f"impute/{dataname}/{exp_name}"
            trainer.test_impute(
                args.impute_condition, 
                imputed_sample_save_dir,
            )
        else:
            trainer.test()
    else:
        ## Save config
        config_save_path = raw_config['model_save_path']
        with open (os.path.join(config_save_path, 'config.pkl'), 'wb') as f:
            pickle.dump(raw_config, f)
        trainer.run_loop()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of TabDiff')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'