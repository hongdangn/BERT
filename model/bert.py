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
        self.linear = nn.Linear(model_dim, 2)
        self.softmax = nn.LogSoftmax(dim = -1) 
    
    def forward(self, output):
        return self.softmax(self.linear(output[:, 0]))
    
class BERTLM(nn.Module):
    def __init__(self, 
                 bert: BERT,
                 model_dim,
                 vocab_size):
        
        self.bert = bert
        self.nsp = BERTNextSentencePrediction(model_dim)
        self.mlm = BERTMaskLM(model_dim, vocab_size)

    def forward(self, input, segment_label):
        output = self.bert(input, segment_label)
        return self.nsp(output), self.mlm(output)