import numpy as np
import torch
import torch.nn as nn
from .base import Problem


class Net(Problem):
    def __init__(self, d=10, n=2, seed=0, q=0.5):
        super().__init__(d, n)
        torch.manual_seed(seed)

        kernel_size = 5
        self.f = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size, stride=kernel_size, bias=False),
            nn.Linear(d//kernel_size, 1)
        )
        self.c = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size, stride=kernel_size, bias=False),
            nn.Linear(d//kernel_size, 1),
            nn.Sigmoid()
        )
        self.q = q

    def target(self, i):
        x = torch.tensor(np.array(i)).to(torch.float32).reshape(-1, 1, self.d)
        y = self.f(x).flatten().detach().numpy()
        return y
    
    def constraints(self, i):
        x = torch.tensor(np.array(i)).to(torch.float32).reshape(-1, 1, self.d)
        y = self.c(x).flatten().detach().numpy() < self.q
        return y