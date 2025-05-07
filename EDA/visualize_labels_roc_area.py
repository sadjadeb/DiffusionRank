from utils import calculate_metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

dataname = 'MQ2007'
exp_name = 'non_learnable_schedule'

discriminative_path = f'../discriminative/experiments/ltr.{dataname}.k1.0.best.results.txt'
real_path = f'../generative/TabDiff/data/{dataname}/test.csv'
info_path = f'../generative/TabDiff/data/{dataname}/info.json'

encoder = OneHotEncoder()
real_data = pd.read_csv(real_path)
real_target = real_data['46'].to_numpy().reshape(-1,1)
real_y = encoder.fit_transform(real_target).toarray()

discriminative_results = {}
discriminative_y = []
discriminative_y_prob = []
with open(discriminative_path) as f:
    lines = f.readlines()
    for line in lines[1:]: # Skip the header line
        qid, true_label, pred_label = line.strip().split()
        qid = int(qid)
        if qid not in discriminative_results:
            discriminative_results[qid] = []
        discriminative_results[qid].append((float(true_label), float(pred_label)))
        
        if true_label == '2':
            true_label = '1'
        
        discriminative_y.append(float(pred_label))
        discriminative_y_prob.append((1 - float(pred_label), float(pred_label)))

discriminative_y = np.array(discriminative_y)
discriminative_y_prob = np.array(discriminative_y_prob)

generative_y = []
for i in range(50):
    generative_path = f'../generative/TabDiff/impute/{dataname}/{exp_name}/{i}.csv'
    generative_data = pd.read_csv(generative_path)
    target = generative_data['46'].to_numpy().reshape(-1, 1)
    generative_y.append(encoder.transform(target).toarray())

generative_y_prob = np.stack(generative_y).mean(0)
generative_y_oh = np.argmax(generative_y_prob, axis=1)
num_classes = np.max(generative_y_oh) + 1
generative_y_oh = np.eye(num_classes)[generative_y_oh]

y_true = real_y.argmax(axis=1)
y_pred = generative_y_prob[:, 1]

generative_results = {}
for idx, label_t, label_p in zip(real_data['47'], y_true, y_pred):
    if idx not in generative_results:
        generative_results[idx] = []
    generative_results[idx].append((label_t, label_p))

discriminative_avgndcg, discriminative_avgp = calculate_metrics(discriminative_results)
print(f"Discriminative - Avg NDCG: {discriminative_avgndcg:.4f} - Avg Precision: {discriminative_avgp:.4f}")
discriminative_auc = roc_auc_score(real_y, discriminative_y_prob, average='micro')
print("Discriminative - AUC: ", round(discriminative_auc*100, 3))

generative_avgndcg, generative_avgp = calculate_metrics(generative_results)
print(f"Generative - Avg NDCG: {generative_avgndcg:.4f} - Avg Precision: {generative_avgp:.4f}")
generative_auc = roc_auc_score(real_y, generative_y_prob, average='micro')
print("Generative - AUC: ", round(generative_auc*100, 3))

# Micro-averaged ROC curves
fpr_gen, tpr_gen, _ = roc_curve(real_y.ravel(), generative_y_prob.ravel())
fpr_disc, tpr_disc, _ = roc_curve(real_y.ravel(), discriminative_y_prob.ravel())

plt.figure(figsize=(8, 6))
plt.plot(fpr_gen, tpr_gen, label=f'Generative (AUC = {generative_auc:.3f}) (NDCG = {generative_avgndcg:.3f})')
plt.plot(fpr_disc, tpr_disc, label=f'Discriminative (AUC = {discriminative_auc:.3f}) (NDCG = {discriminative_avgndcg:.3f})')
# Diagonal line for reference
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')

plt.title("Micro-Averaged ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(f'output/roc_curve_{dataname}.png')

# Plotting ROC curves for each class
for i in range(num_classes):
    fpr_gen, tpr_gen, _ = roc_curve(real_y[:, i], generative_y_prob[:, i])
    fpr_disc, tpr_disc, _ = roc_curve(real_y[:, i], discriminative_y_prob[:, i])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_gen, tpr_gen, label=f'Generative (AUC = {roc_auc_score(real_y[:, i], generative_y_prob[:, i]):.3f}) (NDCG = {generative_avgndcg:.3f})')
    plt.plot(fpr_disc, tpr_disc, label=f'Discriminative (AUC = {roc_auc_score(real_y[:, i], discriminative_y_prob[:, i]):.3f}) (NDCG = {discriminative_avgndcg:.3f})')

    # Diagonal line for reference
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')

    plt.title(f"ROC Curve for Class {i}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f'output/roc_curve_class_{i}_{dataname}.png')
    plt.close()
    
# write the results to a tsv file like qid, true_label, generative_label, discriminative_label
with open(f'output/roc_results_{dataname}.tsv', 'w') as f:
    f.write('qid\ttrue_label\tgenerative_label\tdiscriminative_label\n')
    for idx, t_label, g_label, d_label in zip(real_data['47'], y_true, generative_y_prob[:, 1], discriminative_y_prob[:, 1]):
        f.write(f'{idx}\t{t_label}\t{g_label}\t{d_label}\n')
