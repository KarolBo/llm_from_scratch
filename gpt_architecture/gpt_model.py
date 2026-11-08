import torch

from gpt_config import GPTConfig
from transformer_block import TransformerBlock
from layer_norm import LayerNorm

class GPTModel(torch.nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.token_embeddings = torch.nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embeddings = torch.nn.Embedding(config.context_length, config.embedding_dim)
        self.dropout = torch.nn.Dropout(config.dropout)
        
        self.transformer_blocks = torch.nn.Sequential(
            *[
                DummyTransformerBlock(config) 
                for _ in range(config.n_layers)
            ]
        )

        self.final_norm = LayerNorm(config.embedding_dim)
        self.output_layer = torch.nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

    def forward(self, x):
        batch_size, context_length = x.shape
        
        token_embeddings = self.token_embeddings(x)
        # position_embeddings = self.position_embeddings(
        #     torch.arange(context_length, device=x.device)
        # )
        position_embeddings = self.position_embeddings.weight[:context_length]

        x = token_embeddings + position_embeddings
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.output_layer(x)
        
        return logits