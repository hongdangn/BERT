import os
import sys
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('./tokenizer/bert-vocab.txt', local_files_only=True)

class BaseConfig:
    def __init__(self) -> None:
        super().__init__()

        self.vocab_size: int = len(tokenizer.vocab)
        self.model_dim: int = 256
        self.hidden_size: int = self.model_dim * 4
        self.num_heads: int = 8
        self.num_enc_layers: int = 6

        # model
        self.batch_size: int = 128
        self.num_epochs: int = 50
        self.n_warmup_steps: int = 25
        self.lr: int = 0.005
        self.dropout: float = 0.1
