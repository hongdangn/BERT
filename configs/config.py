class BaseConfig:
    def __init__(self) -> None:
        super().__init__()

        self.vocab_size: int
        self.model_dim: int
        self.hidden_size: int
        self.dropout: float
        self.num_heads: int
        self.num_enc_layers: int

        