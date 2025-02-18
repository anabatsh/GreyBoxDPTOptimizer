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
        state_dim: int,
        action_dim: int,
        seq_len: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        attention_dropout: float = 0.1,
        residual_dropout: float = 0.1,
        embedding_dropout: float = 0.1,
        with_alibi: bool = False,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.seq_norm = nn.LayerNorm(2 * state_dim + action_dim + 1)
        self.seq_proj = nn.Sequential(
            nn.Linear(2 * state_dim + action_dim + 1, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

        self.emb_dropout = nn.Dropout(embedding_dropout)
        self.transformer = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                residual_dropout=residual_dropout,
                max_seq_len=seq_len+1,
                with_alibi=with_alibi,
            )
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, action_dim),
        )
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
            query_state: torch.Tensor,  # [batch_size, state_dim]
            states: torch.Tensor,       # [batch_size, seq_len, state_dim]
            actions: torch.Tensor,      # [batch_size, seq_len] or [batch_size, seq_len, action_dim]
            next_states: torch.Tensor,  # [batch_size, seq_len, state_dim]
            rewards: torch.Tensor,      # [batch_size, seq_len]
        ) -> torch.Tensor:
            if query_state.ndim < 3:
                query_state = query_state.unsqueeze(1)
            if rewards.ndim < 3:
                rewards = rewards.unsqueeze(-1)
            if actions.ndim == 2:
                actions = F.one_hot(actions, num_classes=self.action_dim)

            B, _, D = query_state.size()
            D = max(D, actions.size(-1))
            zeros = torch.zeros((B, 1, D), device=states.device, dtype=states.dtype)
            state_seq = torch.cat([query_state, states], dim=1)
            action_seq = torch.cat([zeros[..., :actions.size(-1)], actions], dim=1)
            next_state_seq = torch.cat([zeros[..., :next_states.size(-1)], next_states], dim=1)
            reward_seq = torch.cat([zeros[..., :1], rewards], dim=1)

            # [batch_size, seq_len + 1, 2 * state_dim + action_dim + 1]
            sequence = torch.cat([state_seq, action_seq, next_state_seq, reward_seq], dim=-1)

            sequence = self.seq_proj(self.seq_norm(sequence))
            out = self.emb_dropout(sequence)

            for block in self.transformer:
                out = block(out)

            out = self.action_head(self.out_norm(out))

            return out
