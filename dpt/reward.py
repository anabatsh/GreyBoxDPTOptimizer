import torch


def relative_improvement(x, y):
    return (x - y) / (torch.finfo(x.dtype).eps + torch.abs(x))

class Reward():
    def __init__(self):
        self.alpha = 0.5

    def offline(self, states, actions, next_states):
        y = states[..., -1]
        y_next = next_states[..., -1]
        cummin = y.cummin(1).values
        cummax = y.cummax(1).values
        scale = torch.maximum(cummax - cummin, torch.tensor(1.))
        rewards = (cummin - y_next) / scale
        rewards = rewards.sign() * rewards.abs().log1p()
        # rewards = relative_improvement(torch.cummin(y, dim=1)[0], y_next).tanh()
        # Reward-To-Go
        return rewards.cumsum(1)

    def online(self, states, actions, next_states):
        y = states[..., -1]
        y_next = next_states[..., -1]
        
        cur_min = torch.min(y, dim=1).values
        cur_max = torch.max(y, dim=1).values
        scale = torch.maximum(cur_max - cur_min, torch.tensor(1.))
        rewards = (cur_min - y_next[:, -1]) / scale
        rewards = rewards.sign() * rewards.abs().log1p()
        # rewards = relative_improvement(y.min(dim=1).values, y_next[:, -1]).tanh()
        return rewards