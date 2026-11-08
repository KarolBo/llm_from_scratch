import torch


class LayerNorm(torch.nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.eps = 1e-5

        self.scale = torch.nn.Parameter(
            torch.ones(embedding_dim)
        )
        self.shift = torch.nn.Parameter(
            torch.zeros(embedding_dim)
        )

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(variance + self.eps) # właściwa normalizacja
        x = x * self.scale + self.shift # application if trainable parameters
        return x