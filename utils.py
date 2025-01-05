import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn


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