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
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.seq_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim + action_dim + 1, 4 * hidden_dim),
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
            )
            for _ in range(num_layers)
        ])
        self.action_head = nn.Linear(hidden_dim, action_dim)
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
        query_state: torch.Tensor,  # [batch_size, state_dim]
        states: torch.Tensor,       # [batch_size, seq_len, state_dim]
        actions: torch.Tensor,      # [batch_size, seq_len]
        next_states: torch.Tensor,  # [batch_size, seq_len, state_dim]
        rewards: torch.Tensor,      # [batch_size, seq_len]
    ) -> torch.Tensor:

        # [batch_size, 1, num_states]
        if query_state.ndim < 3:
            query_state = query_state.unsqueeze(1)

        assert (
            query_state.shape[0] ==
            states.shape[0] ==
            next_states.shape[0] ==
            actions.shape[0] ==
            rewards.shape[0]
        )
        # [batch_size, 1, state_dim] -> [batch_size, 1, hidden_dim]
        query_state_emb = self.state_proj(query_state)
        # [batch_size, seq_len, state_dim] -> [batch_size, seq_len, hidden_dim]
        states_emb = self.state_proj(states)
        # [batch_size, seq_len, state_dim] -> [batch_size, seq_len, hidden_dim]
        next_states_emb = self.state_proj(next_states)
        # [batch_size, seq_len, 1] -> [batch_size, seq_len, action_dim]
        actions_emb = F.one_hot(actions, num_classes=self.action_dim)

        # [batch_size, seq_len + 1, hidden_dim]
        state_seq = torch.cat([query_state_emb, states_emb], dim=1)
        # [batch_size, seq_len + 1, action_dim]
        action_seq = torch.cat(
            [
                torch.zeros(
                    (actions_emb.shape[0], query_state_emb.shape[1], actions_emb.shape[-1]),
                    dtype=actions_emb.dtype,
                    device=actions_emb.device,
                ),
                actions_emb,
            ],
            dim=1,
        )
        # [batch_size, seq_len + 1, hidden_dim]
        next_state_seq = torch.cat(
            [
                torch.zeros(
                    (next_states_emb.shape[0], query_state_emb.shape[1], next_states_emb.shape[-1]),
                    dtype=next_states_emb.dtype,
                    device=next_states_emb.device,
                ),
                next_states_emb,
            ],
            dim=1,
        )
        # [batch_size, seq_len + 1, 1]
        reward_seq = torch.cat(
            [
                torch.zeros(
                    (rewards.shape[0], query_state_emb.shape[1]),
                    dtype=rewards.dtype,
                    device=rewards.device,
                ),
                rewards,
            ],
            dim=1,
        ).unsqueeze(-1)
        # [batch_size, seq_len + 1, 2 * hidden_dim + action_dim + 1]
        sequence = torch.cat([state_seq, action_seq, next_state_seq, reward_seq], dim=-1)

        # [batch_size, seq_len + 1, hidden_dim]
        sequence = self.seq_proj(sequence)

        # [batch_size, seq_len + 1, hidden_dim]
        out = self.emb_dropout(sequence)

        # [batch_size, seq_len + 1, hidden_dim]
        for block in self.transformer:
            out = block(out)

        # [batch_size, seq_len + 1, action_dim]
        out = self.action_head(out)

        return out
