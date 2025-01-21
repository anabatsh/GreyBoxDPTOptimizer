import torch
from torch import nn
from torch.nn import functional as F

try:
    from nn import TransformerBlock
except ImportError:
    from .nn import TransformerBlock


class DPT(nn.Module):
    def __init__(
        self,
        seq_len: int = 3,
        input_dim: int = 1, # размерность х, размерность y = 1
        output_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        attention_dropout: float = 0.1,
        residual_dropout: float = 0.1,
        embedding_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_projector = nn.Linear(input_dim + 1, hidden_dim)
        self.output_projector = nn.Linear(hidden_dim, output_dim)

        self.emb_dropout = nn.Dropout(embedding_dropout)
        self.transformer = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                residual_dropout=residual_dropout,
                max_seq_len=seq_len,
            )
            for _ in range(num_layers)
        ])
    #     self.apply(self._init_weights)

    # @staticmethod
    # def _init_weights(module: nn.Module):
    #     if isinstance(module, (nn.Linear, nn.Embedding)):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #         if isinstance(module, nn.Linear) and module.bias is not None:
    #             torch.nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.LayerNorm):
    #         torch.nn.init.zeros_(module.bias)
    #         torch.nn.init.ones_(module.weight)

    def forward(
        self,
        x: torch.Tensor,  # [batch_size, seq_len, input_dim]
        y: torch.Tensor,  # [batch_size, seq_len]
    ) -> torch.Tensor:

        # [batch_size, seq_len, input_dim]
        x_emb = x
        # [batch_size, seq_len, 1]
        y_emb = y.unsqueeze(-1)

        # [batch_size, seq_len + 1, input_dim]
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

        # [batch_size, seq_len + 1, input_dim + 1]
        sequence = torch.cat([x_seq, y_seq], dim=-1)

        # [batch_size, seq_len + 1, hidden_dim]
        sequence = self.input_projector(sequence)

        # [batch_size, seq_len + 1, hidden_dim]
        out = self.emb_dropout(sequence)

        # [batch_size, seq_len + 1, hidden_dim]
        for block in self.transformer:
            out = block(out)

        # [batch_size, seq_len + 1, output_dim]
        out = self.output_projector(out)
        return out
