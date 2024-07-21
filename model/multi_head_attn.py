import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 num_heads, 
                 model_dim,
                 dropout):
        super(MultiHeadAttention, self).__init__()

        assert model_dim % num_heads == 0

        self.dim = model_dim // num_heads
        self.model_dim = model_dim
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList([
            nn.Linear(self.model_dim, self.model_dim)
                            for _ in range(3)
        ])
        self.out_linear = nn.Linear(self.model_dim, self.model_dim)

    def split_to_heads(self, input):
        batch_size, seq_length, _ = input.size()
        splits = input.view(batch_size, seq_length, self.num_heads, self.dim)

        return splits.transpose(1, 2).contiguous()
    
    def forward(self, input, mask = None):
        """
            input: (batch_size, seq_len, model_dim)
        """
        key, query, val = [
            self.split_to_heads(linear(input))  
                    for linear in self.linear_layers 
        ]

        attn_scores = torch.matmul(key, query.transpose(-2, -1)) / math.sqrt(self.dim)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_scores = F.softmax(attn_scores, dim = -1)

        if self.dropout is not None:
            attn_scores = self.dropout(attn_scores)
        
        output = torch.matmul(attn_scores, val) # (batch, heads, seq_len, dim)
        batch_size, _, seq_length, _ = output.size()
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)

        return self.out_linear(output)  

        
        

