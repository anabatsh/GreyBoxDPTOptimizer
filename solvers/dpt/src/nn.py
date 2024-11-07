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


# WARN: flash attention does not have cpu kernel implementation. You can use torch implementation on cpu, but remember
# that results from flash and torch will be different for the same input.
# example: https://github.com/Dao-AILab/flash-attention/issues/383
class FlashAliBiCausalSelfAttention(nn.Module):
    def __init__(
        self, hidden_dim, num_heads, dropout=0.0, normalize_qk=False, with_alibi=True
    ):
        super().__init__()
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        if with_alibi:
            self.register_buffer(
                "alibi_slopes",
                torch.as_tensor(get_alibi_slopes(num_heads)),
                persistent=False,
            )
        else:
            self.alibi_slopes = None

        if normalize_qk:
            self.q_norm = nn.LayerNorm(hidden_dim // num_heads)
            self.k_norm = nn.LayerNorm(hidden_dim // num_heads)

        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.normalize_qk = normalize_qk

    def forward(self, x, k_cache=None, v_cache=None, cache_seqlens=None):
        B, L, D = x.size()
        # (batch_size, seq_len, 3, num_heads, head_dim)
        qkv = self.in_proj(x).reshape(B, L, 3, self.num_heads, D // self.num_heads)

        # normalizing q,k, see: https://arxiv.org/abs/2302.05442
        if self.normalize_qk:
            q, k, v = qkv.unbind(2)
            q_norm, k_norm = self.q_norm(q), self.k_norm(k)
            qkv = torch.stack([q_norm, k_norm, v], dim=2).to(qkv.dtype)

        # (batch_size, seq_len, num_heads, head_dim)
        if k_cache is None or v_cache is None or cache_seqlens is None:
            out = flash_attn.flash_attn_qkvpacked_func(
                qkv=qkv,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
                alibi_slopes=self.alibi_slopes,
            )
        else:
            assert not self.training
            q, k, v = qkv.unbind(2)
            out = flash_attn.flash_attn_with_kvcache(
                q=q,
                k=k,
                v=v,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=cache_seqlens,
                causal=True,
                alibi_slopes=self.alibi_slopes,
            )
        # (batch_size, seq_len, hidden_dim)
        out = self.out_proj(out.reshape(B, L, D))
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        max_seq_len: int,
        attention_dropout: float,
        residual_dropout: float,
        normalize_qk: bool = False,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = CausalSelfAttentionWithCache(
            hidden_dim,
            num_heads,
            attention_dropout,
            max_seq_len=max_seq_len + 1,
            normalize_qk=normalize_qk,
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(residual_dropout),
        )
        self.pre_norm = pre_norm

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def forward(self, x, k_cache=None, v_cache=None, cache_seqlens=None):
        if self.pre_norm:
            attention_out = self.attention(
                self.norm1(x),
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=cache_seqlens,
            )
            x = x + self.drop(attention_out)
            x = x + self.mlp(self.norm2(x))
        else:
            attention_out = self.attention(
                x, k_cache=k_cache, v_cache=v_cache, cache_seqlens=cache_seqlens
            )
            x = self.norm1(x + self.drop(attention_out))
            x = self.norm2(x + self.mlp(x))

        return x


# WARN: these modules are just an examples of attention implementation from scratch
# they are only for educational purposes here!
def get_alibi_relative_positions(seq_len):
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    return (x - y).to(torch.float)


class AliBiCausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, max_seq_len):
        super().__init__()
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(p=dropout)

        # causal mask
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        # creating alibi attention bias matrix
        alibi_slopes = torch.tensor(get_alibi_slopes(num_heads)).view(
            1, num_heads, 1, 1
        )
        alibi_bias = get_alibi_relative_positions(max_seq_len).view(
            1, 1, max_seq_len, max_seq_len
        )

        alibi_bias = alibi_slopes * alibi_bias
        alibi_bias = alibi_bias.masked_fill(causal_mask == 0, float("-inf"))
        self.register_buffer("alibi_bias", alibi_bias, persistent=False)

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

        # attn
        attn = (query @ key.transpose(-2, -1)) * (1 / math.sqrt(key.size(-1)))
        attn = attn + self.alibi_bias[:, :, :L, :L]
        # [B, nH, L, hD]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ value
        # [B, nH, L, hD] -> [B, L, nH, hD] ->
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        return out


class AliBiCausalSelfAttentionWithCache(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, max_seq_len):
        super().__init__()
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(p=dropout)

        # causal mask
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        # creating alibi attention bias matrix
        alibi_slopes = torch.tensor(get_alibi_slopes(num_heads)).view(
            1, num_heads, 1, 1
        )
        alibi_bias = get_alibi_relative_positions(max_seq_len).view(
            1, 1, max_seq_len, max_seq_len
        )

        alibi_bias = alibi_slopes * alibi_bias
        alibi_bias = alibi_bias.masked_fill(causal_mask == 0, float("-inf"))
        self.register_buffer("alibi_bias", alibi_bias, persistent=False)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

    def forward(self, x, cache=None):
        B, L, D = x.size()

        # [B, L, D * 3] -> 3 * [B, L, D]
        query, key, value = self.in_proj(x).split(self.hidden_dim, dim=-1)
        # [B, L, D] -> [B, nH, L, hD]
        query = query.reshape(B, L, self.num_heads, D // self.num_heads).transpose(1, 2)
        key = key.reshape(B, L, self.num_heads, D // self.num_heads).transpose(1, 2)
        value = value.reshape(B, L, self.num_heads, D // self.num_heads).transpose(1, 2)

        if cache is not None:
            assert L == 1, "with cache input sequence should be length of one"
            key_cache, value_cache = cache
            assert key_cache.shape[0] == value_cache.shape[0] == B
            key = torch.concatenate([key_cache, key], dim=2)
            value = torch.concatenate([value_cache, value], dim=2)

        # attn
        attn = (query @ key.transpose(-2, -1)) * (1 / math.sqrt(key.size(-1)))
        if cache is not None:
            alibi_bias = self.alibi_bias[
                :, :, key.size(-2) - 1, : key.size(-2)
            ].unsqueeze(2)
            attn = attn + alibi_bias
        else:
            attn = attn + self.alibi_bias[:, :, :L, :L]
        # [B, nH, L, hD]
        out = self.attn_drop(F.softmax(attn, dim=-1)) @ value
        # [B, nH, L, hD] -> [B, L, nH, hD] ->
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        return out, (key, value)


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, max_seq_len):
        super().__init__()
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(p=dropout)

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))[None, None, ...]
        self.register_buffer("causal_mask", causal_mask, persistent=False)

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

        # attn
        attn = (query @ key.transpose(-2, -1)) * (1 / math.sqrt(key.size(-1)))
        attn = attn.masked_fill(self.causal_mask[:, :, :L, :L] == 0, float("-inf"))
        # [B, nH, L, hD]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ value
        # [B, nH, L, hD] -> [B, L, nH, hD] ->
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        return out


class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_seq_len: int = 5000):
        super().__init__()
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim)
        )
        pe = torch.zeros(1, max_seq_len, hidden_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_enc", pe, persistent=False)

    def forward(self, x):
        # [batch_size, seq_len, embedding_dim]
        x = x + self.pos_enc[:, : x.size(1)]
        return x


class CausalSelfAttentionWithCache(nn.Module):
    def __init__(
        self, hidden_dim, num_heads, dropout, max_seq_len, normalize_qk, prefix_size=100
    ):
        super().__init__()
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = dropout
        self.normalize_qk = normalize_qk

        if normalize_qk:
            self.q_norm = nn.LayerNorm(hidden_dim // num_heads)
            self.k_norm = nn.LayerNorm(hidden_dim // num_heads)

        # causal mask
        self.causal_mask_with_prefix = torch.tril(
            torch.ones(max_seq_len, max_seq_len, device=DEVICE)
        )
        # self.causal_mask_with_prefix[:100, :100] = 1

        # causal_alibi_mask_with_prefix = torch.tril(
        #     torch.ones(1, num_heads, max_seq_len, max_seq_len)
        # )
        # causal_alibi_mask_with_prefix[..., :, :prefix_size] = 1
        # # creating alibi attention bias matrix
        # alibi_slopes = torch.tensor(get_alibi_slopes(num_heads)).view(
        #     1, num_heads, 1, 1
        # )
        # alibi_bias = get_alibi_relative_positions(max_seq_len - prefix_size).view(
        #     1, 1, max_seq_len - prefix_size, max_seq_len - prefix_size
        # )
        #
        # alibi_bias = alibi_slopes * alibi_bias
        # causal_alibi = torch.tril(
        #     torch.ones(max_seq_len - prefix_size, max_seq_len - prefix_size)
        # )
        # alibi_bias = alibi_bias.masked_fill(causal_alibi == 0, float("-inf"))

        # causal_alibi_mask_with_prefix[..., prefix_size:, prefix_size:] = alibi_bias
        # self.register_buffer(
        #     "causal_alibi_mask_with_prefix",
        #     causal_alibi_mask_with_prefix,
        #     persistent=False,
        # )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

    def forward(
        self, x, k_cache=None, v_cache=None, hidden_cache=None, cache_seqlens=None
    ):
        B, L, D = x.size()

        # [B, L, D * 3] -> 3 * [B, L, D]
        query, key, value = self.in_proj(x).split(self.hidden_dim, dim=-1)
        # [B, L, D] -> [B, nH, L, hD]
        query = query.reshape(B, L, self.num_heads, D // self.num_heads).transpose(1, 2)
        key = key.reshape(B, L, self.num_heads, D // self.num_heads).transpose(1, 2)
        value = value.reshape(B, L, self.num_heads, D // self.num_heads).transpose(1, 2)

        if self.normalize_qk:
            query, key = self.q_norm(query), self.k_norm(key)

        if k_cache is not None and v_cache is not None and cache_seqlens is not None:
            assert L == 1, "with cache input sequence should be length of one"
            assert k_cache.shape[0] == v_cache.shape[0] == B
            # update cache
            # batch_size, max_seq_len, num_heads, head_dim
            k_cache[:, cache_seqlens, :, :] = key.squeeze(2)
            v_cache[:, cache_seqlens, :, :] = value.squeeze(2)
            hidden_cache[:, cache_seqlens, :] = x.squeeze(1)

            if cache_seqlens > 0:
                key = torch.concatenate(
                    [k_cache[:, :cache_seqlens].transpose(1, 2), key], dim=2
                )
                value = torch.concatenate(
                    [v_cache[:, :cache_seqlens].transpose(1, 2), value], dim=2
                )

        # attn
        # attn = (query @ key.transpose(-2, -1)) * (1 / math.sqrt(key.size(-1)))
        if k_cache is not None and v_cache is not None:
            out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=self.causal_mask_with_prefix,
                dropout_p=self.attn_drop,
            )
        else:
            # attn = attn + self.alibi_bias[:, :, :L, :L]
            out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=self.causal_mask_with_prefix[
                    ..., : query.size(2), : query.size(2)
                ],
                dropout_p=self.attn_drop,
            )
        # [B, nH, L, hD]
        # out = self.attn_drop(F.softmax(attn, dim=-1)) @ value
        # [B, nH, L, hD] -> [B, L, nH, hD] ->
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        return out
