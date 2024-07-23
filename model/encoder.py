import torch
import torch.nn as nn
from .multi_head_attn import MultiHeadAttention
from .feed_forward import FeedForward
from .embedding import EmbedLayer

class Encoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_heads,
                 model_dim,
                 hidden_size,
                 dropout):
        super(Encoder, self).__init__()

        assert model_dim % num_heads == 0

        self.multi_head_attn = MultiHeadAttention(num_heads, model_dim, dropout)
        self.feed_forward = FeedForward(model_dim, hidden_size, dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, input: torch.tensor, mask):
        """
            input: (batch_size, seq_length, model_dim)
            encoder mask: (batch_size, 1, 1, seq_length)
        """
        output = self.multi_head_attn(input, mask)
        output = self.layer_norm(output + input)

        final_output = self.feed_forward(output)
        final_output = self.layer_norm(final_output + output)

        return final_output

