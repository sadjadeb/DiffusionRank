from scipy import stats
import argparse
from sklearn.metrics import ndcg_score
from utils import calculate_metrics

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--run_file1", type=str, help="Path to the run file", required=True)
parser.add_argument("--run_file2", type=str, help="Path to the run file", required=True)
args = parser.parse_args()

with open(args.run_file1) as f1, open(args.run_file2) as f2:
    r1_lines = f1.readlines()[1:]
    r2_lines = f2.readlines()[1:]
    
    r1_qids = [line.strip().split()[0] for line in r1_lines]
    r2_qids = [line.strip().split()[0] for line in r2_lines]
    
    r1_true = [float(line.strip().split()[1]) for line in r1_lines]
    r2_true = [float(line.strip().split()[1]) for line in r2_lines]

    r1_preds = [float(line.strip().split()[2]) for line in r1_lines]
    r2_preds = [float(line.strip().split()[2]) for line in r2_lines]

assert r1_true == r2_true, "The true labels in both run files do not match."
assert r1_qids == r2_qids, "The query IDs in both run files do not match."

r1_res = {}
for qid, t, p in zip(r1_qids, r1_true, r1_preds):
    if qid not in r1_res:
        r1_res[qid] = []
    r1_res[qid].append((t, p))

r2_res = {}
for qid, t, p in zip(r2_qids, r2_true, r2_preds):
    if qid not in r2_res:
        r2_res[qid] = []
    r2_res[qid].append((t, p))

print(f"Overall NDCG@10 for Run 1:{calculate_metrics(r1_res)[0]:.6f}")
print(f"Overall NDCG@10 for Run 2:{calculate_metrics(r2_res)[0]:.6f}")


r1_ndcgs = []
r2_ndcgs = []
for qid in r1_res:
    if len(r1_res[qid]) <= 1:
        continue
    
    true_labels = [t[0] for t in r1_res[qid]]
    r1_predicted_scores = [t[1] for t in r1_res[qid]]
    r2_predicted_scores = [t[1] for t in r2_res[qid]]
    
    r1_ndcg = ndcg_score([true_labels], [r1_predicted_scores], k=10)
    r2_ndcg = ndcg_score([true_labels], [r2_predicted_scores], k=10)
    
    r1_ndcgs.append(r1_ndcg)
    r2_ndcgs.append(r2_ndcg)

# Perform the t-test
t_stat, p_value = stats.ttest_rel(r1_ndcgs, r2_ndcgs)
print("T-statistic:", t_stat)
print("P-value:", p_value)

# Significance level
alpha = 0.05
if p_value <= alpha:
    print("The null hypothesis is rejected. The two runs have significantly different predictions.")
else:
    print("The null hypothesis is not rejected. The two runs do not have significantly different predictions.")
