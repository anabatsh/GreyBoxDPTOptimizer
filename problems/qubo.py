from typing import Iterable
import numpy as np
from .base import Problem
import torch

# def int2base(x, d, n):
#     """
#     Convert a given decimal x to a vector of length d in base-n system
#     """
#     if not isinstance(x, Iterable):
#         x = [x]
#     return np.array([list(map(int, list(np.base_repr(i, n, d)))) for i in x], dtype=int)

def int2base(x, d, n):
    """
    Convert a given decimal x to a vector of length d in base-n system
    """
    i = []
    for _ in range(d):
        i.append(x % n)
        x = x // n
    i = np.array(i)[::-1].T
    return i

class QUBO(Problem):
    """
    Quadratic Binary Optimization Problem: x^TQx -> min_x
    """
    def __init__(self, d=1, n=2, seed=0, lazy=True, target=None, **kwargs):
        """
        Additional Input:
            seed - (int) random seed to determine Q
        """
        super().__init__(d, n)
        np.random.seed(seed)
        self.Q = np.triu(np.random.randn(d, d))
        self.info = {"x_min": None, "y_min": target}

        if not lazy:
            x = int2base(np.arange(0, n ** d), d, n)
            y = self.target(x, device='cpu')
            i_best = y.argmin().item()
            self.info = {"x_min": x[i_best], "y_min": y[i_best]}
            del x


    def target(self, x_in, device='cpu'):
        x = torch.tensor(x_in.copy(), device=device, dtype=torch.float32)
        xdim = x.dim()
        if xdim < 2:
            x = x.unsqueeze(0)
        Q = torch.tensor(self.Q, device=device, dtype=torch.float32)
        y = ((x @ Q) * x).sum(1)
        del x, Q
        if xdim < 2:
            y = y.item()
            return y
        return y.detach().cpu().numpy()
