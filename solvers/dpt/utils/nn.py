import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import flash_attn
except ImportError:
    warnings.warn("Missing FlashAttention Install", category=Warning)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    def __init__(
        self, hidden_dim, num_heads, dropout, max_seq_len, prefix_size=1
    ):
        super().__init__()
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = dropout
        self.attn_drop_2 = nn.Dropout(p=dropout)

        # causal mask
        self.causal_mask_with_prefix = torch.tril(
            torch.ones(max_seq_len, max_seq_len, device=DEVICE)
        )[None, None, :, :]
        self.causal_mask_with_prefix[:, :, prefix_size:, :prefix_size] = 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

    def forward(self, x):
        B, L, D = x.size()

        # [B, L, D * 3] -> 3 * [B, L, D]
        query, key, value = self.in_proj(x).split(self.hidden_dim, dim=-1)
        # [B, L, D] -> [B, nH, L, hD]
        query = query.reshape(B, L, self.num_heads, D // self.num_heads).transpose(1, 2)
        key = key.reshape(B, L, self.num_heads, D // self.num_heads).transpose(1, 2)
        value = value.reshape(B, L, self.num_heads, D // self.num_heads).transpose(1, 2)

        # print(f'x: {torch.all(x[:, 0, :] == 0)} | shape: {x.shape}')
        # print(f'Q: {torch.all(query[:, :, 0, :] == 0)} | shape: {query.shape}')
        # print(f'K: {torch.all(key[:, :, 0, :] == 0)} | shape: {key.shape}')
        # print(f'V: {torch.all(value[:, :, 0, :] == 0)} | shape: {value.shape}')

        attn = (query @ key.transpose(-2, -1)) * (1 / math.sqrt(key.size(-1)))
        # print(f'attn zeros: {torch.all(attn[:, :, 0, :] == 0) and torch.all(attn[:, :, :, 0] == 0)} | shape: {attn.shape}')

        attn = attn.masked_fill(self.causal_mask_with_prefix[:, :, :L, :L] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        # print(f'attn masks: {torch.all((attn == 0) == (self.causal_mask_with_prefix == 0))} | sum: {torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)))}')

        out = self.attn_drop_2(attn) @ value
        # print(f'out: {torch.all(out[:, :, 0, :] == 0)} | shape: {out.shape}')

        # out = F.scaled_dot_product_attention(
        #     query, key, value,
        #     attn_mask=self.causal_mask_with_prefix[..., :query.size(2), :query.size(2)],
        #     dropout_p=self.attn_drop,
        # )

        # [B, nH, L, hD] -> [B, L, D]
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        # print(f'out: {torch.all(out[:, 0, :] == 0)} | shape: {out.shape}')
        # print(torch.round(torch.mean(out, 0), decimals=4))
        # print()
        return out