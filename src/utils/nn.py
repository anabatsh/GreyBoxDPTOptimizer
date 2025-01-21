import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        max_seq_len: int,
        attention_dropout: float,
        residual_dropout: float,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = CausalSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            max_seq_len=max_seq_len + 1,
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(residual_dropout),
        )
        self.pre_norm = pre_norm

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def forward(self, x):
        if self.pre_norm:
            attention_out = self.attention(self.norm1(x))
            x = x + self.drop(attention_out)
            x = x + self.mlp(self.norm2(x))
        else:
            attention_out = self.attention(x)
            x = self.norm1(x + self.drop(attention_out))
            x = self.norm2(x + self.mlp(x))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, max_seq_len, prefix_size=1):
        super().__init__()
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = dropout
        self.attn_drop_2 = nn.Dropout(p=dropout)

        # causal mask
        self.causal_mask_with_prefix = torch.tril(
            torch.ones(max_seq_len, max_seq_len)
        )[None, None, :, :]
        # self.causal_mask_with_prefix[:, :, prefix_size:, :prefix_size] = 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

    def forward(self, x):
        B, L, D = x.size()
        query, key, value = self.in_proj(x).split(self.hidden_dim, dim=-1)
        query = query.reshape(B, L, self.num_heads, D // self.num_heads).transpose(1, 2)
        key = key.reshape(B, L, self.num_heads, D // self.num_heads).transpose(1, 2)
        value = value.reshape(B, L, self.num_heads, D // self.num_heads).transpose(1, 2)

        attn = (query @ key.transpose(-2, -1)) * (1 / (key.size(-1) ** 0.5))
        attn = attn.masked_fill(
            (self.causal_mask_with_prefix[:, :, :L, :L] == 0).to(x.device),
            float("-inf")
        )
    
        attn = F.softmax(attn, dim=-1)
        out = self.attn_drop_2(attn) @ value
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)

        return out
