import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
from sklearn.metrics import ndcg_score


def set_all_seeds(seed = 42):
    # Set global random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Ensure deterministic CUDA operations
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    

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
    total_map = 0
    for qid, labels in lables_tuples.items():
        if len(labels) <= 1:
            continue
        
        # Extract true labels and predicted scores
        true_labels = [t[0] for t in labels]
        predicted_scores = [t[1] for t in labels]

        # Sort based on predicted scores in descending order to calculate MAP@k
        sorted_indices = sorted(range(len(predicted_scores)), key=lambda i: predicted_scores[i], reverse=True)
        top_k_indices = sorted_indices[:k]

        # Average Precision at k (AP@k)
        # AP is the average of precision values at each position where a relevant document occurs
        num_relevant = 0
        sum_precisions = 0.0
        for rank, idx in enumerate(top_k_indices, start=1):
            if true_labels[idx] > 0:  # relevant document
                num_relevant += 1
                precision_at_rank = num_relevant / rank
                sum_precisions += precision_at_rank
        
        # Average Precision for this query
        if num_relevant > 0:
            ap_at_k = sum_precisions / num_relevant
        else:
            ap_at_k = 0.0

        # NDCG@k
        ndcg_at_k = ndcg_score([true_labels], [predicted_scores], k=k)

        total_ndcg += ndcg_at_k
        total_map += ap_at_k

    # Calculate averages
    avgndcg = total_ndcg / len(lables_tuples)
    avgmap = total_map / len(lables_tuples)
    
    return avgndcg, avgmap