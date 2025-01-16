import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
from sklearn.metrics import ndcg_score


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, maximize=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.maximize = maximize  # Set to True for metrics where higher is better (like NDCG)

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.maximize:
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        
        return self.early_stop


def calculate_metrics(lables_tuples, k=10):
    total_ndcg = 0
    total_precision = 0
    for qid, labels in lables_tuples.items():
        if len(labels) <= 1:
            continue
        
        # Extract true labels and predicted scores
        true_labels = [t[0] for t in labels]
        predicted_scores = [t[1] for t in labels]

        # Sort based on predicted scores in descending order to calculate P@10
        sorted_indices = sorted(range(len(predicted_scores)), key=lambda i: predicted_scores[i], reverse=True)
        top_k_indices = sorted_indices[:k]

        # Precision at k
        relevant_at_k = sum(1 for i in top_k_indices if true_labels[i] > 0)
        precision_at_k = relevant_at_k / k

        # NDCG@k
        ndcg_at_k = ndcg_score([true_labels], [predicted_scores], k=k)

        total_ndcg += ndcg_at_k
        total_precision += precision_at_k

    # Calculate averages
    avgndcg = total_ndcg / len(lables_tuples)
    avgp = total_precision / len(lables_tuples)
    
    return avgndcg, avgp