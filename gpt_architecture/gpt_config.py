from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int
    context_length: int
    embedding_dim: int
    n_heads: int
    n_layers: int
    dropout: float
    qkv_bias: bool
