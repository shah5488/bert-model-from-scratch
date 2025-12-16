import torch.nn as nn
from model.self_attention import SelfAttention


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()

        self.attention = SelfAttention(hidden_size, num_heads)
        self.attn_norm = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size)
        )

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask):
        attn_out = self.attention(x, attention_mask)
        x = self.attn_norm(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + self.dropout(ffn_out))

        return x
