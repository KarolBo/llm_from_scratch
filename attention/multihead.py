from torch.nn import Linear, Dropout, Module
from torch import triu, ones, bool, inf, softmax


class MultiheadAttention(Module):
    def __init__(self, d_in, d_out, context_length, dropout, n_heads):
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"

        self.dropout = dropout
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads

        self.query = Linear(d_in, d_out, bias=False)
        self.key = Linear(d_in, d_out, bias=False)
        self.value = Linear(d_in, d_out, bias=False)

        self.out_proj = Linear(d_out, d_out)
        self.dropout = Dropout(dropout)

        self.register_buffer(
            'mask',
            triu(
                ones(context_length, context_length, dtype=bool),
                diagonal=1
            )
        )

    def forward(self, x):
        batch_size, context_length, d_in = x.shape
        
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # Split into heads. d_out = n_heads * head_dim
        queries = queries.view(batch_size, context_length, self.n_heads, self.head_dim)
        keys = keys.view(batch_size, context_length, self.n_heads, self.head_dim)
        values = values.view(batch_size, context_length, self.n_heads, self.head_dim)

        # Transpose from (b, ctx, n_heads, head_dim) to (b, n_heads, ctx, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2) # można od razu keys.permute(0, 2, 3, 1)
        values = values.transpose(1, 2)

        # Attention scores
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask[:context_length, :context_length]
        attn_scores.masked_fill_(mask_bool, -inf)

        # Attention weights
        attn_weights = softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Context vectors
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(
            batch_size, context_length, self.head_dim * self.n_heads
        )
        context_vec = self.out_proj(context_vec)

        return context_vec