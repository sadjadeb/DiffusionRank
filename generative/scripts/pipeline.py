import shutil
import os
import argparse
from train import train
from sample import sample
import zero
import lib
import torch
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--sample', action='store_true',  default=False)
    args = parser.parse_args()

    raw_config = lib.load_config(args.config)
    device = torch.device(raw_config['device'])
    
    timer = zero.Timer()
    timer.run()

    if args.train:
        print('Training diffusion model...')
        raw_config['parent_dir'] = raw_config['parent_dir'].format(time.strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(raw_config['parent_dir'], exist_ok=True)
        train(
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
        )
        lib.dump_config(raw_config, os.path.join(raw_config['parent_dir'], 'config.toml'))
    
    if args.sample:
        print('Sampling from diffusion model...')
        sample(
            num_samples=raw_config['sample']['num_samples'],
            batch_size=raw_config['sample']['batch_size'],
            disbalance=raw_config['sample'].get('disbalance', None),
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_path=os.path.join(raw_config['parent_dir'], 'model.pt'),
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            seed=raw_config['seed'],
        )

    print(f'Elapsed time: {str(timer)}')

if __name__ == '__main__':
    main()