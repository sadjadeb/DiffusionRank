import torch
import torch.nn as nn



class DNN(torch.nn.Module):
    def __init__(self, input_dim: int, num_hidden_layers: int = 3, num_hidden_nodes: int = 128, dropout_rate: float = 0.2, approach='pointwise'):
        super(DNN, self).__init__()
        layers = []
        last_dim = input_dim
        
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(last_dim, num_hidden_nodes))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(num_hidden_nodes))
            layers.append(nn.Dropout(p=dropout_rate))
            last_dim = num_hidden_nodes
        layers.append(nn.Linear(last_dim, 1))
        
        if approach == 'pointwise':
            layers.append(nn.ReLU())
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    