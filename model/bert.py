import torch
import torch.nn as nn
from .encoder import Encoder
from .embedding import EmbedLayer
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))

from configs import BaseConfig

class BERTLM(nn.Module):
    def __init__(self, 
                 vocab_size,
                 model_dim,
                 num_heads,
                 num_enc_layers,
                 hidden_size,
                 dropout):
        super(BERTLM, self).__init__()

        self.embed = EmbedLayer(vocab_size, model_dim)
        self.encoder_layers = nn.ModuleList([
            Encoder(vocab_size, num_heads, model_dim, hidden_size, dropout)
                            for layer in range(num_enc_layers)
        ])
        self.mlm = BERTMaskLM(model_dim, vocab_size)
        self.nsp = BERTNextSentencePrediction(model_dim)

    def forward(self, 
                input: torch.tensor, # (batch, seq_len)
                segment_label: torch.tensor):
        mask = (input > 0).unsqueeze(dim = 1).unsqueeze(dim = 1)
        embeded_input = self.embed(input, segment_label)

        for layer in self.encoder_layers:
            embeded_input = layer(embeded_input, mask)

        return self.mlm(embeded_input), self.nsp(embeded_input)
    
class BERTMaskLM(nn.Module):
    def __init__(self, 
                 model_dim,
                 vocab_size):
        super(BERTMaskLM, self).__init__()

        self.linear = nn.Linear(model_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim = -1)

    def forward(self, output):
        return self.softmax(self.linear(output))
    
class BERTNextSentencePrediction(nn.Module):
    def __init__(self, model_dim):
        super(BERTNextSentencePrediction, self).__init__()
        
        self.linear = nn.Linear(model_dim, 2)
        self.softmax = nn.LogSoftmax(dim = -1) 
    
    def forward(self, output):
        return self.softmax(self.linear(output[:, 0]))
    
if __name__ == "__main__":
    cfg = BaseConfig()
    model = BERTLM(
        vocab_size = cfg.vocab_size,
        model_dim = cfg.model_dim,
        num_heads = cfg.num_heads,
        num_enc_layers = cfg.num_enc_layers,
        hidden_size = cfg.hidden_size,
        dropout = cfg.dropout
    )

    input = torch.randint(0, cfg.vocab_size, (cfg.batch_size, 5))
    segment_label = torch.randint(1, 3, (cfg.batch_size, 5))

    x = torch.randn((cfg.batch_size, 5, cfg.model_dim))
    print(model(input, segment_label))
