class BaseConfig:
    def __init__(self) -> None:
        super().__init__()

        self.vocab_size: int = 100
        self.model_dim: int = 20
        self.hidden_size: int = 40
        self.num_heads: int = 4
        self.num_enc_layers: int = 3

        # model
        self.batch_size: int = 2
        self.dropout: float = 0.1
