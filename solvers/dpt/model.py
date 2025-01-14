from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F

try:
    from utils.nn import TransformerBlock
except ImportError:
    from .utils.nn import TransformerBlock

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DPT_K2D(nn.Module):
    def __init__(
        self,
        num_actions: int,
        seq_len: int = 200,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        attention_dropout: float = 0.5,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.1,
        pre_norm: bool = True
    ) -> None:
        super().__init__()

        self.num_actions = num_actions
        # [action + next_state]
        self.embed_transition = nn.Linear(num_actions + 1, hidden_dim)

        self.emb_dropout = nn.Dropout(embedding_dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                residual_dropout=residual_dropout,
                pre_norm=pre_norm,
                max_seq_len=seq_len,
            )
            for _ in range(num_layers)
        ])
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        x: torch.Tensor,  # [batch_size, seq_len]
        y: torch.Tensor,  # [batch_size, seq_len]
    ) -> torch.Tensor:

        assert y.shape[0] == x.shape[0]

        # [batch_size, seq_len, num_actions]
        x_emb = F.one_hot(x, num_classes=self.num_actions)
        # [batch_size, seq_len, 1]
        y_emb = y.unsqueeze(-1)

        # [batch_size, seq_len + 1, num_actions]
        x_seq = torch.cat(
            [
                torch.zeros(
                    (x_emb.shape[0], 1, x_emb.shape[-1]),
                    dtype=x_emb.dtype,
                    device=x_emb.device,
                ),
                x_emb,
            ],
            dim=1,
        )
        # [batch_size, seq_len + 1, 1]
        y_seq = torch.cat(
            [
                torch.zeros(
                    (y_emb.shape[0], 1, y_emb.shape[-1]),
                    dtype=y_emb.dtype,
                    device=y_emb.device,
                ),
                y_emb,
            ],
            dim=1,
        )

        # [batch_size, seq_len + 1, num_actions + 1]
        sequence = torch.cat([x_seq, y_seq], dim=-1)

        # [batch_size, seq_len + 1, hidden_dim]
        sequence = self.embed_transition(sequence)

        # [batch_size, seq_len + 1, hidden_dim]
        out = sequence #self.emb_dropout(sequence)

        # [batch_size, seq_len + 1, hidden_dim]
        for block in self.blocks:
            out = block(out)

        # [batch_size, seq_len + 1, num_actions]
        out = self.action_head(out)

        # [batch_size, num_actions]
        if not self.training:
            return out[:, -1, :]

        # [batch_size, seq_len + 1, num_actions]
        return out