import numpy as np
import torch
import torch.nn as nn
from .base import Problem


class ReLUNet(Problem):
    """
    Optimization problem that has a simple ReLU 
    neural network as a target and constraints function.
    """
    def __init__(self, d=10, n=2, seed=0):
        """
        Additional Input:
            seed - random seed to determine the nets (int)
            q - rate of the points satisfying the constraints (float in [0, 1])
        """
        super().__init__(d, n)
        torch.manual_seed(seed)

        width = 2 * d
        self.f = nn.Sequential(
            nn.Linear(d, width), nn.ReLU(),
            # nn.Linear(width, width), nn.ReLU(),
            # nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, 1)
        )
        # depth = 4
        # hidden_dims = [d] + [width] * depth + [1]
        # self.f = nn.ModuleList()
        # for i in range(len(hidden_dims) - 1):
        #     self.f.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        #     self.f.append(nn.ReLU())
        # self.f.append(nn.Linear(hidden_dims[-1], 1))
        self.name = f'ReLUNet_{seed}'

    def target(self, i):
        x = torch.tensor(np.array(i)).to(torch.float32).reshape(-1, self.d)
        y = self.f(x).flatten().detach().numpy()
        return y
    
    def constraints(self, i):
        return torch.ones(i.shape[0]).to(torch.bool)
    
class ConvNet(Problem):
    """
    Optimization problem that has a simple convolutional 
    neural network as a target and constraints function.
    """
    def __init__(self, d=10, n=2, seed=0, q=0.5):
        """
        Additional Input:
            seed - random seed to determine the nets (int)
            q - rate of the points satisfying the constraints (float in [0, 1])
        """
        super().__init__(d, n)
        torch.manual_seed(seed)

        kernel_size = min(5, d)
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
        self.name = f'ConvNet_{seed}'

    def target(self, i):
        x = torch.tensor(np.array(i)).to(torch.float32).reshape(-1, 1, self.d)
        y = self.f(x).flatten().detach().numpy()
        return y
    
    def constraints(self, i):
        x = torch.tensor(np.array(i)).to(torch.float32).reshape(-1, 1, self.d)
        y = self.c(x).flatten().detach().numpy() < self.q
        return y