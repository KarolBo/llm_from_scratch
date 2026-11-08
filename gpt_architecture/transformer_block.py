import torch

from gpt_config import GPTConfig


class TransformerBlock(torch.nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

    def forward(self, x):
        return x