import torch
import torch.nn as nn



class DNN(torch.nn.Module):
    def __init__(self, input_dim: int, approach: str, num_hidden_layers: int = 4, num_hidden_nodes: int = 128, dropout_rate: float = 0.2):
        super(DNN, self).__init__()
        layers = []
        last_hidden_dim = input_dim
        last_dim = 2 if approach == 'pointwise' else 1
        
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(last_hidden_dim, num_hidden_nodes))
            layers.append(nn.SiLU())
            layers.append(nn.LayerNorm(num_hidden_nodes))
            layers.append(nn.Dropout(p=dropout_rate))
            last_hidden_dim = num_hidden_nodes
        layers.append(nn.Linear(last_hidden_dim, last_dim))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
