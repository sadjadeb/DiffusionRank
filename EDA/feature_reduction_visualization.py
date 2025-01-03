import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

# Generate a sample dataset with 8 features
np.random.seed(42)

dataset_name = 'MSLR-Web10K'
clustering_method = 'dbscan'
scaling_method = 'minmax'
transposition = False


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
train_data_path = os.path.join(project_root, 'data', dataset_name, 'npy', 'Fold1', 'X_num_train.npy')
train_label_path = os.path.join(project_root, 'data', dataset_name, 'npy', 'Fold1', 'y_train.npy')
test_data_path = os.path.join(project_root, 'data', dataset_name, 'npy', 'Fold1', 'X_num_test.npy')
test_label_path = os.path.join(project_root, 'data', dataset_name, 'npy', 'Fold1', 'y_test.npy')


X_train = np.load(train_data_path)
y_train = np.load(train_label_path)

X_train = np.concatenate((y_train.reshape(-1, 1), X_train), axis=1)
X_train_transposed = X_train.T

# Normalize data
if scaling_method == 'standard':
    scaler = StandardScaler()
elif scaling_method == 'minmax':
    scaler = MinMaxScaler()
    
if transposition:
    data_normalized = scaler.fit_transform(X_train_transposed)
else:
    data_normalized = scaler.fit_transform(X_train)

print(data_normalized.shape)

# Dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
data_tsne = tsne.fit_transform(data_normalized)

pca = PCA(n_components=2, random_state=42)
data_pca = pca.fit_transform(data_normalized)

svd = TruncatedSVD(n_components=2, random_state=42)
data_svd = svd.fit_transform(data_normalized)

# DBSCAN clustering
dbscan_tsne = DBSCAN(eps=4.5, min_samples=80).fit_predict(data_tsne)
dbscan_pca = DBSCAN(eps=0.5, min_samples=80).fit_predict(data_pca)
dbscan_svd = DBSCAN(eps=0.5, min_samples=80).fit_predict(data_svd)

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# t-SNE plot
axs[0].set_title("t-SNE with DBSCAN Clustering")
unique_labels = set(dbscan_tsne)
for label in unique_labels:
    color = 'black' if label == -1 else plt.cm.rainbow(float(label) / len(unique_labels))
    label_name = 'Noise' if label == -1 else f'Cluster {label}'
    cluster_points = data_tsne[dbscan_tsne == label]
    axs[0].scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color], label=label_name, s=20)

# PCA plot
axs[1].set_title("PCA with DBSCAN Clustering")
unique_labels = set(dbscan_pca)
for label in unique_labels:
    color = 'black' if label == -1 else plt.cm.rainbow(float(label) / len(unique_labels))
    label_name = 'Noise' if label == -1 else f'Cluster {label}'
    cluster_points = data_pca[dbscan_pca == label]
    axs[1].scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color], label=label_name, s=20)

# SVD plot
axs[2].set_title("SVD with DBSCAN Clustering")
unique_labels = set(dbscan_svd)
for label in unique_labels:
    color = 'black' if label == -1 else plt.cm.rainbow(float(label) / len(unique_labels))
    label_name = 'Noise' if label == -1 else f'Cluster {label}'
    cluster_points = data_svd[dbscan_svd == label]
    axs[2].scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color], label=label_name, s=20)

# Highlighting the label point
if transposition:
    axs[0].scatter(data_tsne[0, 0], data_tsne[0, 1], c='red', s=100, edgecolor='black')
    axs[0].text(data_tsne[0, 0], data_tsne[0, 1], 'Label', fontsize=12, fontweight='bold', color='red', ha='right')
    
    axs[1].scatter(data_pca[0, 0], data_pca[0, 1], c='red', s=100, edgecolor='black')
    axs[1].text(data_pca[0, 0], data_pca[0, 1], 'Label', fontsize=12, fontweight='bold', color='red', ha='right')

    axs[2].scatter(data_svd[0, 0], data_svd[0, 1], c='red', s=100, edgecolor='black')
    axs[2].text(data_svd[0, 0], data_svd[0, 1], 'Label', fontsize=12, fontweight='bold', color='red', ha='right')


# Adjusting layout and showing plot
for ax in axs:
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", fontsize='small')
plt.tight_layout()
plt.savefig(f'output/tsne_{clustering_method}_clusters_{dataset_name.lower()}_{scaling_method}_{"transposed" if transposition else "normal"}.png')
