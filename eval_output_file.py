import argparse
from utils import calculate_metrics
from sklearn.metrics import roc_auc_score
import numpy as np

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--run_file", type=str, help="Path to the run file")
args = parser.parse_args()

results = {}
true_labels = []
pred_labels = []
with open(args.run_file) as f:
    lines = f.readlines()
    for line in lines[1:]:  # Skip the header line
        qid, true_label, pred_label = line.strip().split()
        if qid not in results:
            results[qid] = []
        results[qid].append((float(true_label), float(pred_label)))

        if true_label == '2':
            true_label = '1'

        true_labels.append((1 - float(true_label), float(true_label)))
        pred_labels.append((1 - float(pred_label), float(pred_label)))

avgndcg, avgp = calculate_metrics(results)

true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)
auc = roc_auc_score(true_labels, pred_labels, average='micro')

print(f'avgp: {avgp:.6f}, avgndcg: {avgndcg:.6f}, auc: {auc:.6f}')
