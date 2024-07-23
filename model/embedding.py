import torch
import torch.nn as nn
import math

class EmbedLayer(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 model_dim):
        
        super(EmbedLayer, self).__init__()

        self.vocab_size = vocab_size
        self.model_dim = model_dim

        self.segm_embed = nn.Embedding(3, model_dim, padding_idx = 0)
        self.token_embed = nn.Embedding(vocab_size, model_dim, padding_idx = 0)
    
    def positional_embed(self, input_tensor):
        """
            input_tensor: (batch_size, seq_length)
        """
        batch_size, seq_length = input_tensor.size()

        def pos_embed(index: int):
            embed = torch.zeros(self.model_dim)
            embed = [math.sin(index / (10000 ** (id / self.model_dim))) if id % 2 == 0 else 
                        math.cos(index / (10000 ** ((id - 1) / self.model_dim))) for id in range(self.model_dim)]

            return torch.tensor(embed)

        for id in range(batch_size):
            for index in range(seq_length): # sequence id
                if index == 0:
                    embed_seq = pos_embed(index).unsqueeze(dim = 0)
                else:
                    emb_tok = pos_embed(index).unsqueeze(dim = 0)
                    embed_seq = torch.cat((embed_seq, emb_tok), dim = 0)
            if id == 0:
                embed_batch = embed_seq.unsqueeze(dim = 0)
            else:
                embed_batch = torch.cat((embed_batch, embed_seq.unsqueeze(dim = 0)), dim = 0)

        return embed_batch

    def forward(self, input_tensor, segment_label):
        segments_embed = self.segm_embed(segment_label)
        pos_embed = self.positional_embed(input_tensor)
        tokens_embed = self.token_embed(input_tensor)

        return segments_embed + pos_embed + tokens_embed

if __name__ == "__main__":
    x = torch.randint(0, 10, (5, 3))
    segment_label = torch.randint(1, 3, (5, 3))
    print(EmbedLayer(10, 5)(x, segment_label).size())