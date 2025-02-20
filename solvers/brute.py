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

def get_grid(d, n, n_nodes=1024):
    """
    For given d and n list all the vectors [{1, n}]^d
    *if there's too many, list only len_max of them with equal step
    input: d, n - integer
    input: i - either all or a part of the vectors [{1, n}]^d
    """
    d_new = min(d, int(np.emath.logn(n, n_nodes)))
    x = np.arange(0, n ** d_new)
    i = int2base(x, d_new, n)
    if d_new < d:
        i = np.pad(i, ((0, 0), (0, d - d_new)), constant_values=0)
        i = np.pad(i, ((0, 1), (0, 0)), constant_values=n-1)
    return i

class BRUTE(Solver):
    def __init__(self, problem, budget, k_init=0, k_samples=0, seed=0):
        super().__init__(problem, budget, k_init, k_samples, seed)

    def optimize(self):
        self.logger = Logger(self)

        device = "cpu"
        x = torch.tensor(
            get_grid(self.problem.d, self.problem.n, self.budget), 
            device=device
        ).float()
        y = self.problem.target(x)
        i_best = torch.argmin(y).item()

        x_min = x[i_best].numpy()
        y_min = y[i_best].numpy()
        self.logger.logs['x_best'] = x_min.tolist()
        self.logger.logs['y_best'] = float(y_min)
        self.logger.logs['y_list'] = y.tolist()
        self.logger.logs['m_list'] = list(range(len(y)))
        return self.logger.logs