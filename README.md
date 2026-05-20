# DiffusionRank: Learning to Rank via Denoising Diffusion

This repository contains the official implementation of **DiffusionRank**, a diffusion-based deep generative approach to Learning-to-Rank (LTR) that models the joint distribution over feature vectors and relevance labels.

## Overview

Traditional Learning-to-Rank methods use discriminative machine learning approaches that model the probability of document relevance given query-document features. DiffusionRank takes an alternative approach by extending [TabDiff](https://github.com/MinkaiXu/TabDiff), a denoising diffusion-based generative model for tabular data, to create generative equivalents of classical discriminative pointwise and pairwise LTR objectives.

![DiffusionRank](DiffusionRank.jpg)

### Key Features

- **Generative LTR**: Models the full joint distribution over features and relevance labels
- **Pointwise & Pairwise**: Supports both pointwise and pairwise training approaches
- **Efficient Inference**: Single-step denoising for relevance prediction, comparable to discriminative models
- **Flexible Architecture**: Uses feedforward networks with minimal parameter overhead

## Repository Structure

```
DiffusionRank/
├── generative/                    # DiffusionRank implementation
│   ├── main.py                    # Main training/testing script
│   ├── tabdiff/                   # TabDiff-based diffusion modules
│   │   ├── models/                # Diffusion model implementations
│   │   ├── modules/               # Neural network modules
│   │   ├── trainer.py             # Training logic
│   │   └── tabdiff_configs.toml   # Model configurations
│   └── synetune_launcher.py       # Hyperparameter tuning
├── discriminative/                # Discriminative baseline models
│   ├── model.py                   # Neural network architecture
│   ├── pointwise.py               # Pointwise discriminative model
│   ├── pairwise.py                # Pairwise discriminative model (RankNet)
│   ├── pointwise_perturbed.py     # Pointwise with perturbed features
│   ├── pairwise_perturbed.py      # Pairwise with perturbed features
│   ├── listwise_lambdarank.py     # LambdaRank implementation
│   └── xgb.py                     # XGBoost baseline
├── EDA/                                  # Exploratory data analysis
├── ltr_dataset_to_numpy.py               # Convert raw LTR data to numpy and create fraction subsets
├── compute_ranking_metrics.py            # Evaluation script
├── ndcg_significance_test.py             # Statistical significance testing
└── utils.py                              # Utility functions
```

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Dependencies

- Python 3.8+
- PyTorch
- NumPy
- scikit-learn
- Weights & Biases (wandb)

## Data Preparation

### Downloading Datasets

<!-- TODO: Add download instructions -->

We evaluate on three standard LTR benchmarks:

| Dataset | Queries (Train/Val/Test) | Features | Relevance Labels |
|---------|--------------------------|----------|------------------|
| MQ2007 | 1,017 / 339 / 336 | 46 | 3 (0-2) |
| MQ2008 | 471 / 157 / 156 | 46 | 3 (0-2) |
| MSLR-WEB10K | 6,000 / 2,000 / 2,000 | 136 | 5 (0-4) |

### Data Format

Place raw LETOR files under `data/{dataset}/raw/`. Then run:

```bash
python ltr_dataset_to_numpy.py --dataset MQ2007 --fold 1
```

This converts raw text to numpy and creates fraction subsets in one step, producing:

```
data/
├── MQ2007/
│   └── by_fraction/
│       └── Fold1/
│           ├── k1.0/
│           │   ├── X_train.npy, y_train.npy, idx_train.npy
│           │   ├── X_train_non.npy, y_train_non.npy, idx_train_non.npy
│           │   ├── X_val.npy, y_val.npy, idx_val.npy
│           │   └── X_test.npy, y_test.npy, idx_test.npy
│           ├── k0.5/
│           │   └── ...
│           ├── k0.25/
│           ├── ...
├── MQ2008/
│   └── ...
└── MSLR-WEB10K/
    └── ...
```

Fraction values: `1.0, 0.5, 0.25, 0.0625, 0.015625, 0.00390625`

## Training

### DiffusionRank (Generative)

#### Pointwise Training

```bash
python generative/main.py \
    --dataname MQ2007 \
    --approach pointwise \
    --mode train \
    --steps 15000 \
    --lr 5e-6 \
    --batch_size 4096 \
    --dim_t 256 \
    --num_layers 4 \
    --gpu 0
```

#### Pairwise Training

```bash
python generative/main.py \
    --dataname MQ2007 \
    --approach pairwise \
    --mode train \
    --steps 15000 \
    --lr 5e-6 \
    --batch_size 4096 \
    --dim_t 256 \
    --num_layers 4 \
    --gpu 0
```

#### Training with Data Fractions

Use the `--k` parameter to train with a subset of data:

```bash
python generative/main.py \
    --dataname MSLR-WEB10K \
    --approach pointwise \
    --mode train \
    --k 0.25 \
    --gpu 0
```

### Discriminative Baselines

#### Pointwise

```bash
python discriminative/pointwise.py \
    --dataset MQ2007 \
    --task train \
    --num_hidden_nodes 256 \
    --lr 5e-6 \
    --k 1.0
```

#### Pairwise (RankNet)

```bash
python discriminative/pairwise.py \
    --dataset MQ2007 \
    --task train \
    --num_hidden_nodes 256 \
    --lr 5e-6 \
    --k 1.0
```

### XGBoost Baseline

```bash
python baselines/xgb.py \
    --dataset MQ2007 \
    --approach pointwise \
    --k 1.0
```

## Testing & Evaluation

### Testing DiffusionRank

```bash
python generative/main.py \
    --dataname MQ2007 \
    --approach pointwise \
    --mode test \
    --ckpt_path checkpoints/MQ2007/your_experiment/best_model.pt \
    --gpu 0
```

### Computing Ranking Metrics

Evaluate predictions using NDCG@10 and MAP@10:

```bash
python compute_ranking_metrics.py --run_file predictions/your_predictions.txt
```

### Statistical Significance Testing

```bash
python ndcg_significance_test.py
```

## Key Arguments

### Generative Model (`generative/main.py`)

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataname` | Dataset name (MQ2007, MQ2008, MSLR-WEB10K, MSLR-WEB30K) | - |
| `--approach` | Training approach (pointwise, pairwise) | pointwise |
| `--mode` | Mode (train, test) | train |
| `--steps` | Number of training steps | 15000 |
| `--lr` | Learning rate | 5e-6 |
| `--batch_size` | Batch size | 4096 |
| `--dim_t` | Hidden dimension | 256 |
| `--num_layers` | Number of hidden layers | 4 |
| `--k` | Fraction of training data to use | 1.0 |
| `--gpu` | GPU index (-1 for CPU) | 0 |
| `--no_wandb` | Disable Weights & Biases logging | False |

### Discriminative Models

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset name | - |
| `--task` | Task (train, test) | - |
| `--num_hidden_nodes` | Hidden layer size | - |
| `--lr` | Learning rate | 5e-6 |
| `--k` | Fraction of training data | 1.0 |
| `--checkpoint` | Path to model checkpoint (for testing) | None |
| `--no_wandb` | Disable Weights & Biases logging | False |

## Model Architecture

DiffusionRank extends the discriminative model architecture by:

1. **Input**: Adding the (possibly masked) relevance label and diffusion time step as additional inputs
2. **Output**: Jointly predicting the relevance label and the noise added to features

The model uses a feedforward network with:
- 4 hidden layers (default)
- SiLU activation
- Layer normalization
- Dropout (0.1)
- Hidden size: 256 (LETOR 4.0) or 1024 (MSLR-WEB10K)

## Results

DiffusionRank demonstrates consistent improvements over discriminative baselines on standard LTR benchmarks, particularly on larger datasets like MSLR-WEB10K. The generative training objective helps the model learn more robust representations by modeling the joint distribution of features and labels.

## Logging

Training progress is logged to [Weights & Biases](https://wandb.ai/). To disable logging:

```bash
python generative/main.py --dataname MQ2007 --approach pointwise --mode train --no_wandb
```

## Acknowledgements

This work builds upon [TabDiff](https://github.com/MinkaiXu/TabDiff), a mixed-type diffusion model for tabular data generation.
