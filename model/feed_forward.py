import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, 
                 model_dim,
                 hidden_size,
                 dropout):
        super(FeedForward, self).__init__()

        self.model_dim = model_dim
        self.hidden_size = hidden_size
        self.in_linear = nn.Linear(model_dim, hidden_size)
        self.out_linear = nn.Linear(hidden_size, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        
    def forward(self, input: torch.tensor):
        input = self.gelu(self.in_linear(input))
        input = self.dropout(input)

        return self.out_linear(input)