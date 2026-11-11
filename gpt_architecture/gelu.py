from torch import nn, tensor, pi, pow, sqrt, tanh


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        gelu = 0.5 * x * (1 + tanh(
            sqrt(tensor(2.0) / pi) * 
            (x + 0.044715 * pow(x, 3))
        ))
        return gelu