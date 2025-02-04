import torch
import numpy as np


def relative_improvement(x, y):
    return torch.abs(x - y) / (torch.finfo(x.dtype).eps + torch.abs(x))

class Reward():
    def __init__(self):
        self.alpha = 0.5
    
    def offline(self, states, actions, next_states):
        return torch.sign(states[..., -1] - next_states[..., -1])
        # y = states[..., -1]
        # y_next = next_states[..., -1]
        # seq_len = actions.shape[-1]
        # reward_exploit = relative_improvement(torch.cummin(y, dim=0)[0], y_next)
        # print(reward_exploit.shape)
        # mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool)).unsqueeze(0)
        # y_expanded = torch.where(mask, y.unsqueeze(0), torch.inf)
        # exploration = torch.abs(y_expanded - y_next.unsqueeze(1))
        # reward_explore = 1 / (1 + exploration.min(dim=1).values)
        # rewards = self.alpha * reward_exploit + (1 - self.alpha) * reward_explore
        # return rewards
    
    def online(self, states, actions, next_states):
        return torch.sign(states[..., -1] - next_states[..., -1])
        # y = next_states[..., -1]
        # y = torch.cat((y[:, [0]], y), dim=1)

        # reward_exploit = relative_improvement(y[:, :-1].min(dim=1).values, y[:, -1])
        # exploration = torch.abs(y[:, :-1] - y[:, -1].unsqueeze(1))
        # reward_explore = 1 / (1 + exploration.min(dim=1).values)
        # reward = self.alpha * reward_exploit + (1 - self.alpha) * reward_explore
        # return reward