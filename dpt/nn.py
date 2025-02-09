import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

# class GeLUMLP(nn.Module):
#     def __init__(self, hidden_size, intermediate_size, bias: bool = False):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.intermediate_size = intermediate_size
#         self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
#         self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
#         self.act_fn = nn.GELU()

#     def forward(self, hidden_state):
#         return self.down_proj(self.act_fn(self.up_proj(hidden_state)))

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
        self.drop1 = nn.Dropout(residual_dropout)
        self.drop2 = nn.Dropout(residual_dropout)

        self.attention = CausalSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            max_seq_len=max_seq_len,
        )
        self.mlp = SwiGLUMLP(hidden_dim, int(7 / 2 * hidden_dim), bias=True)
        self.pre_norm = pre_norm

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def forward(self, x):
        if self.pre_norm:
            x = x + self.drop1(self.attention(self.norm1(x)))
            x = x + self.drop2(self.mlp(self.norm2(x)))
        else:
            attention_out = self.attention(x)
            x = self.norm1(x + self.drop(attention_out))
            x = self.norm2(x + self.mlp(x))
        return x


def get_alibi_slopes(n):
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_alibi_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def get_relative_positions(seq_len: int) -> torch.tensor:
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    return x - y


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, max_seq_len, with_alibi=False, normalize_qk=True):
        super().__init__()
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = dropout
        self.attn_drop_fn = nn.Dropout(p=dropout)

        self.use_sdpa = True

        self.register_buffer(
            "causal_mask", torch.triu(-torch.inf * torch.ones(1, 1, max_seq_len, max_seq_len), diagonal=1)
        )
        self.with_alibi = with_alibi
        if with_alibi:
            self.register_buffer(
                "alibi_slopes",
                torch.as_tensor(get_alibi_slopes(num_heads)),
                persistent=False,
            )
        else:
            self.alibi_slopes = None

        self.normalize_qk = normalize_qk
        if normalize_qk:
            self.q_norm = nn.LayerNorm(hidden_dim // num_heads)
            self.k_norm = nn.LayerNorm(hidden_dim // num_heads)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

    def forward(self, x):
        B, L, D = x.size()
        query, key, value = self.in_proj(x).split(self.hidden_dim, dim=-1)
        # [B, L, D] -> [B, nH, L, hD]
        query = query.reshape(B, L, self.num_heads, D // self.num_heads).transpose(1, 2)
        key = key.reshape(B, L, self.num_heads, D // self.num_heads).transpose(1, 2)
        value = value.reshape(B, L, self.num_heads, D // self.num_heads).transpose(1, 2)

        # normalizing q,k, see: https://arxiv.org/abs/2302.05442
        if self.normalize_qk:
            query, key = self.q_norm(query), self.k_norm(key)

        attn_bias = self.causal_mask[:, :, :L, :L]
        if self.with_alibi:
            attn_bias = attn_bias + (self.alibi_slopes[:, None, None] * get_relative_positions(L)[None, :].to(self.alibi_slopes.device))

        if self.use_sdpa:
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_bias, dropout_p=self.attn_drop if self.training else 0)
        else:
            attn = (query @ key.transpose(-2, -1)) * (1 / (key.size(-1) ** 0.5))
            attn += attn_bias
            attn = F.softmax(attn, dim=-1)
            out = self.attn_drop_fn(attn) @ value
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)

        return out
