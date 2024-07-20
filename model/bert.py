import torch
import torch.nn as nn
from encoder import Encoder
from embedding import EmbedLayer

class BERT(nn.Module):
    def __init__(self, 
                 vocab_size,
                 model_dim,
                 num_heads,
                 num_enc_layers,
                 hidden_size,
                 dropout):
        super(BERT, self).__init__()

        self.embed = EmbedLayer(vocab_size, model_dim)
        self.encoder_layers = nn.Sequential([
            Encoder(vocab_size, num_heads, model_dim, hidden_size, dropout)
                            for layer in range(num_enc_layers)
        ])

    def forward(self, 
                input: torch.tensor, # (batch, seq_len)
                segment_label: torch.tensor):
        mask = (input > 0).unsqueeze(dim = 1).unsqueeze(dim = 1)
        embeded_input = self.embed(input, segment_label)

        return self.encoder_layers(embeded_input, mask)
    