import os
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import QuantileTransformer
import argparse
from utils import set_all_seeds, calculate_metrics

# Set seed
seed = 42
set_all_seeds(seed)

# Parse arguments
parser = argparse.ArgumentParser(description='Run LambdaMART LTR Experiment')
parser.add_argument('--dataset', type=str, default='MQ2007', choices=['MQ2007', 'MQ2008', 'MSLR-WEB10K', 'MSLR-WEB30K'], help='Dataset to use for the experiment')
parser.add_argument('--k', type=float, default=1.0, help='Fraction k for the dataset')
parser.add_argument('--objective', type=str, choices=['regression', 'binary'], default='binary', help='Objective function for LightGBM')
args = parser.parse_args()

dataset = args.dataset
k = args.k
print(f'Running LambdaMART experiment for dataset: {dataset}, k: {k}, objective: {args.objective}')

# Setup hyperparameters
data_normalization = 'quantile'

# Set data paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_root, 'data', dataset, 'by_fraction', 'Fold1', f'k{k}')


X_train = np.load(os.path.join(data_dir, 'X_num_train.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
idx_train = np.load(os.path.join(data_dir, 'idx_train.npy'))

X_val = np.load(os.path.join(data_dir, 'X_num_val.npy'))
y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
idx_val = np.load(os.path.join(data_dir, 'idx_val.npy'))

X_test = np.load(os.path.join(data_dir, 'X_num_test.npy'))
y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
idx_test = np.load(os.path.join(data_dir, 'idx_test.npy'))

# Binarize labels
threshold_of_neg = 1 if 'MSLR' in dataset else 0
y_train[y_train <= threshold_of_neg], y_train[y_train > threshold_of_neg] = 0, 1
y_val[y_val <= threshold_of_neg], y_val[y_val > threshold_of_neg] = 0, 1
y_test[y_test <= threshold_of_neg], y_test[y_test > threshold_of_neg] = 0, 1

# Normalize
if data_normalization == 'quantile':
    normalizer = QuantileTransformer(output_distribution='normal',
                                     n_quantiles=max(min(X_train.shape[0] // 30, 1000), 10),
                                     subsample=int(1e9),
                                     random_state=seed)
    X_train = normalizer.fit_transform(X_train)
    X_val = normalizer.transform(X_val)
    X_test = normalizer.transform(X_test)


# Prepare group/query data (this is required by LightGBM for LTR)
def count_queries_by_id(qids):
    _, counts = np.unique(qids, return_counts=True)
    return counts.tolist()

train_group = count_queries_by_id(idx_train)
val_group = count_queries_by_id(idx_val)
test_group = count_queries_by_id(idx_test)

# Prepare LightGBM datasets
lgb_train = lgb.Dataset(X_train, y_train, group=train_group)
lgb_val = lgb.Dataset(X_val, y_val, group=val_group, reference=lgb_train)

# LambdaMART model params
params = {
    # Objective & metric
    "objective": args.objective,
    "metric": ["binary_logloss", "auc", "ndcg"] if args.objective == 'binary' else ["ndcg"],
    "ndcg_eval_at": [10],
    "boosting_type": "gbdt",
    "verbose": -1,

    # Regularization
    "lambda_l1": 0.1,
    "lambda_l2": 0.2,

    # Tree complexity
    "num_leaves": 128 if 'MSLR' in dataset else 64,    # Controls tree complexity

    # Learning rate & boosting
    "learning_rate": 0.05,           # Conservative learning rate
    "num_iterations": 100,           # Total boosting iterations

    # Sampling
    "feature_fraction": 0.8,          # Feature bagging
    "bagging_fraction": 0.8,          # Data bagging
    "bagging_freq": 5,                # Perform bagging every k iterations

    # System
    "num_threads": 16,                # Adjust based on your CPU
    "seed": 42,                       # Reproducibility
}

# Train
model = lgb.train(params,
                  lgb_train,
                  valid_sets=[lgb_train, lgb_val],
                  valid_names=['train', 'valid'])
print('Training completed.')

# Predict and evaluate
def evaluate_lgb(model, X, true_labels, qids):
    pred_labels = model.predict(X)
    results = {}
    for y_true, y_pred, qid in zip(true_labels, pred_labels, qids):
        if qid not in results:
            results[qid] = []
        results[qid].append((y_true, y_pred))

    avgndcg, avgp = calculate_metrics(results)
    return avgndcg, avgp, results

print('Evaluating on test set...')
avgndcg, avgp, test_results = evaluate_lgb(model, X_test, y_test, idx_test)
print(f'Test NDCG: {avgndcg:.6f}, Test P: {avgp:.6f}')
print(f'{avgndcg:.6f} {avgp:.6f}')

# Save model
model_save_path = os.path.join('checkpoints', f'ltr.{dataset}.lambdamart.k{k}.quantile.model.txt')
model.save_model(model_save_path)
print(f'Model saved to {model_save_path}')

# Save results
results_save_path = os.path.join('predictions', f'ltr.{dataset}.lambdamart.k{k}.quantile.results.txt')
with open(results_save_path, 'w') as f:
    f.write('qid true_label pred_score\n')
    for qid, values in test_results.items():
        for true_label, pred_score in values:
            f.write(f'{qid} {true_label} {pred_score}\n')
print(f'Results saved to {results_save_path}')
