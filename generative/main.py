import numpy as np
import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D, MLP1D, GaussianDiffusionMLP1D
import wandb
import pickle
import os
from utils import set_all_seeds, calculate_metrics

set_all_seeds(42)

device = 'cuda:3'
num_features = 137
training_steps = 500000
objective = 'pred_noise'
mode = 'inpaint' # 'train' or 'inpaint'
inpainting_strategy = 't-noised-replace' # 't-noised-replace' or 'original-replace' or 't-noised-replace'
inpainting_index = 0
dataset_name = 'MSLR-WEB30K' # 'MQ2007' or 'MQ2008' or 'MSLR-WEB10K' or 'MSLR-WEB30K'
base_dir = os.path.join("..", "data", dataset_name, "npy", "Fold1")


def preprocess_tensor(tensor, feature_length, target_channels=32):
    # Determine how many samples should remain (nearest multiple of target_channels)
    n_samples = tensor.shape[0]
    truncate_size = (n_samples // target_channels) * target_channels  # Nearest multiple of target_channels

    # Truncate the tensor to make the number of samples divisible by target_channels
    tensor_truncated = tensor[:truncate_size]

    # Reshape the tensor to (x, target_channels, feature_length)
    reshaped_tensor = tensor_truncated.view(-1, target_channels, feature_length)
    
    return reshaped_tensor

def load_dataset_and_concat(base_dir, split):
    # Load train, test, and validation data
    idx = np.load(os.path.join(base_dir, f"idx_{split}.npy"))
    X_num = np.load(os.path.join(base_dir, f"X_num_{split}.npy"))
    y = np.load(os.path.join(base_dir, f"y_{split}.npy"))
    
    # Concatenate the features and label
    concat_data = np.concatenate([y.reshape(-1, 1), X_num], axis=1)
    
    # cast to tensor for further processing
    concat_tensor = torch.tensor(concat_data, dtype=torch.float32)
    
    return concat_tensor, idx


# Load train, test, and validation data
train_data, train_idx = load_dataset_and_concat(base_dir, "train")
test_data, test_idx = load_dataset_and_concat(base_dir, "test")
val_data, val_idx = load_dataset_and_concat(base_dir, "val")

# preprocess the tensors
train_seq = preprocess_tensor(train_data, num_features)
test_seq = preprocess_tensor(test_data, num_features)
val_seq = preprocess_tensor(val_data, num_features)

print("Train Shape:", train_seq.shape)
print("Test Shape:", test_seq.shape)
print("Val Shape:", val_seq.shape)

# modify sequences shape to run the model temporarily
train_seq = torch.cat([train_seq, torch.zeros(train_seq.shape[0], 32, 1)], dim=-1)
test_seq = torch.cat([test_seq, torch.zeros(test_seq.shape[0], 32, 1)], dim=-1)
val_seq = torch.cat([val_seq, torch.zeros(val_seq.shape[0], 32, 1)], dim=-1)

# model = Unet1D(
#     dim = 64,
#     dim_mults = (1, 2, 4, 8),
#     channels = 32
# )

# diffusion = GaussianDiffusion1D(
#     model,
#     seq_length = num_features + 1, # temporarily added 1 to the feature length
#     objective = objective,
#     auto_normalize = False
# ).to(device)


model = MLP1D(
    input_dim=num_features + 1,
)

diffusion = GaussianDiffusionMLP1D(
    model,
    seq_length=num_features + 1,
    objective=objective,
    auto_normalize=False
).to(device)

if mode == 'train':
    wandb.init(project=f'ddpm_pt_{dataset_name}', name=f'exp_{training_steps}_{objective}_MLP')
    
    wandb.config.update({
        'training_steps': training_steps,
        'objective': objective,
        'mode': mode,
    })

    train_dataset = Dataset1D(train_seq)
    val_dataset = Dataset1D(val_seq)
    
    trainer = Trainer1D(
        diffusion,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_batch_size = 32,
        train_lr = 8e-5,
        train_num_steps = training_steps, # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        eval_every = 10,    # save model and sample every n steps
    )
    
    best_model, final_model = trainer.train()
    
    torch.save(best_model.model.state_dict(), f'pt_experiments/ddpm_{dataset_name}_{training_steps}_{objective}_MLP_best.pt')
    torch.save(final_model.model.state_dict(), f'pt_experiments/ddpm_{dataset_name}_{training_steps}_{objective}_MLP_final.pt')
    
    wandb.finish()

elif mode == 'inpaint':
    # load the trained model
    diffusion.model.load_state_dict(torch.load(f'pt_experiments/ddpm_{dataset_name}_{training_steps}_{objective}_MLP_best.pt'))
    
    # replace the inpainting index with random values
    masked_test_seq = test_seq.clone()
    # original_labels = masked_test_seq[:, :, inpainting_index].unique()
    # random_labels = np.random.choice(original_labels, size=(masked_test_seq.shape[0], masked_test_seq.shape[1],))
    random_labels = np.zeros((masked_test_seq.shape[0], masked_test_seq.shape[1],))
    masked_test_seq[:, :, inpainting_index] = torch.tensor(random_labels, dtype=torch.float32)
    masked_test_seq = masked_test_seq.to(device)
    
    # add noise to the sequence
    noise = torch.randn_like(masked_test_seq, device=device)
    t = torch.full((masked_test_seq.shape[0],), diffusion.num_timesteps - 1, device=device)
    noisy_test_seq = diffusion.q_sample(x_start=masked_test_seq, t=t, noise=noise)
    
    # inpaint the sequence
    test_seq = test_seq.to(device)
    denoised_test_seq = diffusion.inpaint(noisy_test_seq, test_seq, noise, strategy=inpainting_strategy, inpainting_index=inpainting_index)
    denoised_test_seq = denoised_test_seq.detach().cpu()
    test_seq = test_seq.detach().cpu()
    
    # reshape the denoised sequence to (n_samples, 600)
    denoised_test_seq = denoised_test_seq.numpy().reshape(-1, num_features + 1)
    test_seq = test_seq.numpy().reshape(-1, num_features + 1)
    
    y_true = test_seq[:, inpainting_index]
    y_pred = denoised_test_seq[:, inpainting_index]
    
    test_idx = test_idx[:len(denoised_test_seq)]
    
    # calculate the MSE over the inpainted value
    mse = np.mean((y_true - y_pred) ** 2)
    print(f'MSE over index {inpainting_index}: {mse}')
    
    results = {}
    for idx, label_t, label_p in zip(test_idx, y_true, y_pred):
        if idx not in results:
            results[idx] = []
        results[idx].append((label_t, label_p))
    
    avgndcg, avgp = calculate_metrics(results)
    print(f'avgndcg: {avgndcg}, avgp: {avgp}')
    
    y_true_min, y_true_max = np.min(y_true), np.max(y_true)
    y_pred_min, y_pred_max = np.min(y_pred), np.max(y_pred)
    print(f'y_true_max: {y_true_max:.2f}, y_true_min: {y_true_min:.2f}')
    print(f'y_pred_max: {y_pred_max:.2f}, y_pred_min: {y_pred_min:.2f}')
    
    # write the results to a file
    with open(f'pt_experiments/inpaint_{dataset_name}_steps-{training_steps}_obj-{objective}_index-{inpainting_index}_MLP.txt', 'w') as f:
        f.write('qid\ttrue\tpred\n')
        for idx, labels in results.items():
            for label_t, label_p in labels:
                f.write(f'{idx}\t{label_t:.1f}\t{label_p:.4f}\n')
