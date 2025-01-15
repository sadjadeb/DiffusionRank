import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import zero
import lib
from tab_ddpm import GaussianMultinomialDiffusion
from utils_train import make_dataset
from tab_ddpm.modules import MLPDiffusion
import wandb
import argparse
import time
from tqdm import trange


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class Trainer:
    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, val_loader, approach, device=torch.device('cuda:1')):
        self.diffusion = diffusion
        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.val_loader = val_loader
        self.approach = approach
        self.device = device
        self.log_every = 100
        self.print_every = 500

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        if self.approach == 'pointwise':
            loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
        elif self.approach == 'pairwise':
            loss_multi, loss_gauss = self.diffusion.mixed_loss_pairwise(x, out_dict)
        loss = loss_multi + loss_gauss
        
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss

    def run_loop(self):
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        
        best_val_loss = float('inf')
        best_model_state = None

        curr_count = 0
        for step in trange(self.steps):
            x, out_dict = next(self.train_iter)
            out_dict = {'y': out_dict}
            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_every == 0:
                    
                    self.diffusion.eval()
                    
                    val_count = 0
                    val_loss_multi = 0.0
                    val_loss_gauss = 0.0
                    
                    for x_val in self.val_loader:
                        x_val = x_val.to(self.device)
                        if self.approach == 'pairwise':
                            if x_val.shape[0] % 2 == 1:
                                x_val = x_val[:-1]
                        
                        val_count += len(x_val)
                        if self.approach == 'pointwise':
                            loss_multi, loss_gauss = self.diffusion.mixed_loss(x_val, {})
                        elif self.approach == 'pairwise':
                            loss_multi, loss_gauss = self.diffusion.mixed_loss_pairwise(x_val, {})
                        val_loss_multi += loss_multi.item() * len(x_val)
                        val_loss_gauss += loss_gauss.item() * len(x_val)
                        
                    val_mloss = np.around(val_loss_multi / val_count, 4)
                    val_gloss = np.around(val_loss_gauss / val_count, 4)
                    val_loss = val_mloss + val_gloss
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = self.diffusion._denoise_fn.state_dict()
                     
                    
                    self.diffusion.train()
                    
                    # print(f'Step {step}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss} Validation Loss: {val_loss}')
                    # log to wandb
                    wandb.log({'loss': mloss + gloss, 'val_loss': val_loss})
                
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0
                
        return best_model_state


def train(
    parent_dir,
    real_data_path,
    steps = 1000,
    lr = 0.002,
    weight_decay = 1e-4,
    batch_size = 1024,
    model_params = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    num_numerical_features = 0,
    seed = 42,
    approach = 'pointwise',
    device = "cpu",
):
    real_data_path = os.path.normpath(real_data_path)
    parent_dir = os.path.normpath(parent_dir)

    zero.improve_reproducibility(seed)

    T = lib.Transformations(**T_dict)

    dataset = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
    )

    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])

    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features
    model_params['d_in'] = d_in

    train_loader = lib.prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)

    X_val = torch.from_numpy(dataset.X_num["val"]).float()
    val_loader = DataLoader(X_val, batch_size=batch_size, shuffle=False)

    model = MLPDiffusion(**model_params).to(device)
    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device
    )
    diffusion.to(device)
    diffusion.train()

    trainer = Trainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        val_loader=val_loader,
        approach=approach,
        device=device
    )
    best_model_state = trainer.run_loop()

    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.final.pt'))
    torch.save(best_model_state, os.path.join(parent_dir, 'model.best.pt'))

  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--k', type=float)
    parser.add_argument('--approach', type=str, default='pointwise', choices=['pointwise', 'pairwise'])
    args = parser.parse_args()

    raw_config = lib.load_config(args.config)
    device = torch.device(raw_config['device'])
    dataset = raw_config['dataset']
    
    if args.k is not None:
        experiment_id = f"k{args.k}"
        raw_config['real_data_path'] = raw_config['real_data_path'].format(dataset, experiment_id)
    else:
        experiment_id = time.strftime('%Y-%m-%d_%H-%M-%S')
        raw_config['real_data_path'] = raw_config['real_data_path'].format(dataset, "k1.0")
        
    if args.approach == 'pairwise':
        experiment_id += "_pairwise"
    
    raw_config['parent_dir'] = raw_config['parent_dir'].format(dataset, experiment_id)
    os.makedirs(raw_config['parent_dir'], exist_ok=True)
    lib.dump_config(raw_config, os.path.join(raw_config['parent_dir'], 'config.toml'))
    
    wandb.init(project=f"tddpm_{dataset}", name=f"exp_{experiment_id}", config=raw_config)
    
    train(
        **raw_config['train']['main'],
        **raw_config['diffusion_params'],
        parent_dir=raw_config['parent_dir'],
        real_data_path=raw_config['real_data_path'],
        model_params=raw_config['model_params'],
        T_dict=raw_config['train']['T'],
        num_numerical_features=raw_config['num_numerical_features'],
        seed=raw_config['seed'],
        approach=args.approach,
        device=device,
    )
    
    
    wandb.finish()
