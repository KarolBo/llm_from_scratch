import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from torch import nn

from gpt_config import GPTConfig
from feed_forward import FeedForward
from attention.multihead import MultiheadAttention


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.attention = MultiheadAttention(
            d_in=config.embedding_dim,
            d_out=config.embedding_dim,
            context_length=config.context_length,
            n_heads=config.n_heads,
            dropout=config.dropout,
            qkv_bias=config.qkv_bias,
        )

        self.feed_forward = FeedForward(
            embedding_dim=config.embedding_dim,
            hidden_dim_factor=4,   
        )

        self.layer_norm1 = nn.LayerNorm(config.embedding_dim)
        self.layer_norm2 = nn.LayerNorm(config.embedding_dim)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Attention block
        shortcut = x # residual connection
        x = self.layer_norm1(x) # pre-normalization
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut

        # Feed forward block
        shortcut = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + shortcut
        return x