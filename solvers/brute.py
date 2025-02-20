import torch
import numpy as np
from .base import Logger, Solver


def int2base(x, d, n):
    """
    Convert a given decimal x to a vector of length d in base-n system
    """
    i = []
    for _ in range(d):
        i.append(x % n)
        x //= n
    return list(reversed(i))

class BRUTE(Solver):
    def __init__(self, problem, budget, k_init=0, k_samples=0, seed=0):
        super().__init__(problem, budget, k_init, k_samples, seed)

    def optimize(self):
        self.logger = Logger(self)

        device = "cpu"
        x = torch.tensor([
            int2base(i, self.problem.d, self.problem.n) 
            for i in range(self.problem.n ** self.problem.d)
        ], device=device).float()
        y = self.problem.target(x)
        i_best = torch.argmin(y).item()

        x_min = x[i_best].numpy()
        y_min = y[i_best].numpy()
        self.logger.logs['x_best'] = x_min.tolist()
        self.logger.logs['y_best'] = float(y_min)
        self.logger.logs['y_list'] = y.tolist()
        self.logger.logs['m_list'] = list(range(len(y)))
        return self.logger.logs