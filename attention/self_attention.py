from torch.nn import Module, Linear, Dropout, ModuleList
from torch import softmax, triu, ones, bool, inf, cat

class CasualAttention(Module):
    def __init__(self, d_in, d_out, context_length, dropout):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        # self.W_query = Parameter(rand(d_in, d_out))
        # self.W_key = Parameter(rand(d_in, d_out))
        # self.W_value = Parameter(rand(d_in, d_out))
        self.query = Linear(d_in, d_out, bias=False)
        self.key = Linear(d_in, d_out, bias=False)
        self.value = Linear(d_in, d_out, bias=False)
        self.dropout = Dropout(dropout)

        # Buffers are automatically moved to appropriate device (CPU or GPU) along with the model.
        self.register_buffer(
            'mask',
            triu(
                ones(context_length, context_length, dtype=bool),
                diagonal=1
            )
        )

    def forward(self, x):
        batch_size, context_length, d_in = x.shape
        # queries = x @ self.W_query
        # keys = x @ self.W_key
        # values = x @ self.W_value
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        attn_scores = queries @ keys.transpose(1, 2) # keep the batch dimension at 0
        attn_scores.masked_fill_( # trailing underscore -> in-place
            self.mask[:context_length, :context_length],
            -inf
        )

        attn_weights = softmax(
            attn_scores / keys.shape[-1]**0.5, # scaling factor to keep variance close to 1
            dim=-1
        ) 
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


class MultiheadAttentionWrapper(Module):
    def __init__(self, d_in, d_out, context_length, dropout, n_heads):
        super().__init__()
        self.heads = ModuleList([
            CasualAttention(d_in, d_out, context_length, dropout)
            for _ in range(n_heads)
        ])

    def forward(self, x):
        return cat(
            [
                head(x) 
                for head in self.heads
            ],
            dim=-1
        )