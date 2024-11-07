from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from .nn import TransformerBlock

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DPT_K2D(nn.Module):
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        rnn_weights_path: Optional[str] = None,
        seq_len: int = 200,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        state_rnn_embedding: int = 16,
        attention_dropout: float = 0.5,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.1,
        rnn_dropout: float = 0.0,
        normalize_qk: bool = False,
        pre_norm: bool = True,
    ) -> None:
        super().__init__()

        self.num_states = num_states
        self.num_actions = num_actions
        self.seq_len = seq_len

        self.emb_dropout = nn.Dropout(embedding_dropout)

        self.embed_transition = nn.Linear(
            state_rnn_embedding + num_actions + 1, # [state, next_state, action, reward]
            hidden_dim,
        )
        self.embedd = nn.Linear(num_states, state_rnn_embedding)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                    normalize_qk=normalize_qk,
                    pre_norm=pre_norm,
                    max_seq_len=seq_len,
                )
                for _ in range(num_layers)
            ]
        )
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
        query_states: torch.Tensor,         # [batch_size, 1]
        context_states: torch.Tensor,       # [batch_size, seq_len]
        context_next_states: torch.Tensor,  # [batch_size, seq_len]
        context_actions: torch.Tensor,      # [batch_size, seq_len]
        context_rewards: torch.Tensor,      # [batch_size, seq_len]
    ) -> torch.Tensor:

        if query_states.ndim < 2:
            query_states = query_states.unsqueeze(1)

        # print('query_states', query_states.shape)
        # print('context_states', context_states.shape)
        # print('context_actions', context_actions.shape)
        # print('context_rewards', context_rewards.shape)
        # print('-'*20)

        # [batch_size, 1, state_rnn_embedding]
        query_states_emb = self.embedd(query_states.unsqueeze(-1))
        # [batch_size, seq_len, state_rnn_embedding]
        context_states_emb = self.embedd(context_states.unsqueeze(-1))
        # [batch_size, seq_len, num_actions]
        context_actions_emb = F.one_hot(context_actions, num_classes=self.num_actions)

        # print('query_states_emb', query_states_emb.shape)
        # print('context_states_emb', context_states_emb.shape)
        # print('context_actions_emb', context_actions_emb.shape)
        # print('context_rewards', context_rewards.shape)

        if self.training:
            # [batch_size, seq_len]
            shuffle_idx = torch.randint(
                low=0, high=context_actions.size(1), size=(context_actions.size(1),)
            )
            # [batch_size, seq_len, state_rnn_embedding]
            context_states_emb = context_states_emb[:, shuffle_idx]
            # [batch_size, seq_len, num_actions]
            context_actions_emb = context_actions_emb[:, shuffle_idx]
            # [batch_size, seq_len]
            context_rewards = context_rewards[:, shuffle_idx]

        # [batch_size, seq_len + 1, state_rnn_embedding]
        state_seq = torch.cat([query_states_emb, context_states_emb], dim=1)

        # [batch_size, seq_len + 1, num_actions]
        action_seq = torch.cat(
            [
                torch.zeros(
                    (context_actions_emb.size(0), 1, context_actions_emb.size(2)),
                    dtype=context_actions_emb.dtype,
                    device=context_actions_emb.device,
                ),
                context_actions_emb,
            ],
            dim=1,
        )

        # [batch_size, seq_len + 1, 1]
        reward_seq = torch.cat(
            [
                torch.zeros(
                    (context_rewards.size(0), 1),
                    dtype=context_rewards.dtype,
                    device=context_rewards.device,
                ),
                context_rewards,
            ],
            dim=1,
        ).unsqueeze(-1)

        # [batch_size, seq_len + 1, num_actions + 1 + state_rnn_embedding]
        sequence = torch.cat([action_seq, reward_seq, state_seq], dim=-1)
        # [batch_size, seq_len + 1, hidden_dim]
        sequence = self.embed_transition(sequence)

        # [batch_size, seq_len + 1, hidden_dim]
        out = self.emb_dropout(sequence)
        for block in self.blocks:
            out = block(out)
        # [batch_size, seq_len + 1, hidden_dim]

        # [batch_size, seq_len + 1, num_actions]
        out = self.action_head(out)

        # [batch_size, num_actions]
        if not self.training:
            return out[:, -1, :]

        # [batch_size, seq_len + 1, num_actions]
        return out
