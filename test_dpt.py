import numpy as np
import torch
from torch.nn import functional as F

from utils import int2bin, get_xaxis
from solvers.dpt.src.model_dpt import DPT_K2D

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path):
    model = DPT_K2D(
        num_states=1,
        num_actions=16,
        hidden_dim=32,
        seq_len=50,
        num_layers=4,
        num_heads=4,
        attention_dropout=0.5,
        residual_dropout=0.1,
        embedding_dropout=0.3,
        normalize_qk=False,
        pre_norm=True,
        rnn_weights_path=None,
        state_rnn_embedding=1,
        rnn_dropout=0.0,
    ).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint)
    return model

def test(model, problem, n_steps=10):
    temp = 1.0
    do_samples = False

    all_actions = get_xaxis(d=problem.d, n=problem.n)
    all_states = problem.target(all_actions)
    target_state = torch.tensor(all_states.min())

    query_states = torch.tensor([all_states.max()])
    context_states = torch.Tensor(1, 0)
    context_next_states = torch.Tensor(1, 0)
    context_actions = torch.Tensor(1, 0)
    context_rewards = torch.Tensor(1, 0)

    for _ in range(n_steps):
        predicted_actions = model(
            query_states=query_states.to(dtype=torch.float, device=DEVICE),
            context_states=context_states.to(dtype=torch.float, device=DEVICE),
            context_next_states=context_next_states.to(dtype=torch.float, device=DEVICE),
            context_actions=context_actions.to(dtype=torch.long, device=DEVICE),
            context_rewards=context_rewards.to(dtype=torch.float, device=DEVICE),
        )
        temp = 1.0 if temp <= 0 else temp
        probs = F.softmax(predicted_actions / temp, dim=-1)
        if do_samples:
            predicted_action = torch.multinomial(probs, num_samples=1).squeeze(1).cpu()[0]
        else:
            predicted_action = torch.argmax(probs, dim=-1).cpu()[0]

        point = int2bin(predicted_action, d=problem.d, n=problem.n)
        state = torch.tensor(problem.target(point))
        # print(f'step {_} | current target: {query_states.item():>8.6} -> suggested point: {point} -> new target: {target.item():.6}')

        context_states = torch.cat([context_states, query_states.unsqueeze(0)], dim=1)
        context_next_states = torch.cat([context_next_states, state.unsqueeze(0)], dim=1)
        context_actions = torch.cat([context_actions, torch.tensor([predicted_action]).unsqueeze(0)], dim=1)
        context_rewards = torch.cat([context_rewards, (-1 * (state - query_states)).unsqueeze(0)], dim=1)
        query_states = state

    # print()
    # print(f'found minimal value: {target.item():.6}')
    # print(f'ground truth: {ground_truth_state.item():.6}')
    # print()
    # print(f'all possible targets in an order:\n{np.sort(all_states)}')
    print(f"{context_next_states[0].min()} | {target_state}")
    return {
            "states": context_states[0],
            "actions": context_actions[0],
            "next_states": context_next_states[0],
            "rewards": context_rewards[0],
            "target_state": target_state,
            "accuracy": 1 - (context_next_states[0].min() - target_state) / (torch.tensor(all_states.max()) - target_state)
        }