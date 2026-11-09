from torch import nn

from gelu import GELU


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim_factor: int=4):
        super().__init__()

        # It internally expands the embedding dimension into 
        # a higher dimensional space through the first linear layer.
        # It llows exploration of a richer representation space.
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_factor * embedding_dim),
            GELU(),
            nn.Linear(hidden_dim_factor * embedding_dim, embedding_dim),
        )
        # Druga warstwa ma swobodnie projektować wynik z powrotem 
        # do przestrzeni embeddingów, bez dodatkowego ograniczenia 
        # przez funkcję aktywacji

    def forward(self, x):
        return self.layers(x)