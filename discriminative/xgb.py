import os
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import QuantileTransformer
import argparse
from utils import set_all_seeds, calculate_metrics

# Set random seeds for reproducibility
seed = 42
set_all_seeds(seed)

# Parse arguments
parser = argparse.ArgumentParser(description='Run XGBoost Experiment')
parser.add_argument('--dataset', type=str, default='MQ2007', choices=['MQ2007', 'MQ2008', 'MSLR-WEB10K', 'MSLR-WEB30K'], help='Dataset to use for the experiment')
parser.add_argument('--k', type=float, default=1.0, help='Fraction k for the dataset')
parser.add_argument('--approach', type=str, choices=['pointwise', 'pairwise', 'listwise'], default='pointwise', help='Approach for training')
args = parser.parse_args()

dataset = args.dataset
k = args.k
print(f'Running XGBoost experiment for dataset: {dataset}, k: {k}, approach: {args.approach}')

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
if args.approach == 'pointwise':
    threshold_of_neg = 1 if 'MSLR' in dataset else 0
    y_train[y_train <= threshold_of_neg], y_train[y_train > threshold_of_neg] = 0, 1

# Normalize
if data_normalization == 'quantile':
    normalizer = QuantileTransformer(output_distribution='normal',
                                     n_quantiles=max(min(X_train.shape[0] // 30, 1000), 10),
                                     subsample=int(1e9),
                                     random_state=seed)
    X_train = normalizer.fit_transform(X_train)
    X_val = normalizer.transform(X_val)
    X_test = normalizer.transform(X_test)

# Prepare DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

if args.approach == 'pointwise':
    objective = 'binary:logistic'
elif args.approach == 'pairwise':
    objective = 'rank:pairwise'
elif args.approach == 'listwise':
    objective = 'rank:ndcg'
else:
    raise ValueError(f'Unknown approach: {args.approach}')

params = {
    'objective': objective,
    "eval_metric": ["ndcg@10"],
    "tree_method": "hist",
    "seed": seed,
}

if objective == 'rank:pairwise' or objective == 'rank:ndcg':
    params['lambdarank_num_pair_per_sample'] = 10
    params['lambdarank_pair_method'] = "topk"

if 'MSLR' in dataset:
    params['max_depth'] = 32

model = xgb.train(params, dtrain,
                    num_boost_round=3000,
                    evals=[(dval, "valid")],
                    early_stopping_rounds=50,
                    verbose_eval=50)
print("Training completed.")

def evaluate_xgb(model, X, true_labels, qids):
    dtest = xgb.DMatrix(X)
    pred_labels = model.predict(dtest)
    results = {}
    for y_true, y_pred, qid in zip(true_labels, pred_labels, qids):
        if qid not in results:
            results[qid] = []
        results[qid].append((y_true, y_pred))

    avgndcg, avgmap = calculate_metrics(results)
    return avgndcg, avgmap, results

print('Evaluating on test set...')
avgndcg, avgmap, test_results = evaluate_xgb(model, X_test, y_test, idx_test)
print(f'Test NDCG: {avgndcg:.6f}, Test MAP: {avgmap:.6f}')
print(f'{avgndcg:.6f} {avgmap:.6f}')

model_save_path = os.path.join('checkpoints', f'ltr.{dataset}.xgboost.{args.approach}.k{k}.quantile.model.json')
model.save_model(model_save_path)
print(f'Model saved to {model_save_path}')

# Save results
results_save_path = os.path.join('predictions', f'ltr.{dataset}.xgboost.{args.approach}.k{k}.quantile.results.txt')
with open(results_save_path, 'w') as f:
    f.write('qid true_label pred_score\n')
    for qid, values in test_results.items():
        for true_label, pred_score in values:
            f.write(f'{qid} {true_label} {pred_score}\n')
print(f'Results saved to {results_save_path}')