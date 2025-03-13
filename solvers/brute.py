import torch
import numpy as np
from .base import Solver


def int2base(x, d, n):
    """
    Convert a given decimal x to a vector of length d in base-n system
    """
    i = []
    for _ in range(d):
        i.append(x % n)
        x //= n
    i = torch.stack(i).flip(0).T
    return i


def get_grid(d, n):
    """
    For given d and n list all the vectors [{1, n}]^d
    input: d, n - integer
    input: i - vectors [{1, n}]^d
    """
    x = torch.arange(0, n ** d)
    i = int2base(x, d, n)
    return i


class BRUTE(Solver):
    def __init__(self, problem, budget, seed=0):
        assert problem.d <= np.emath.logn(problem.n, 1024), "The dimension of the problem is too high for brute force"
        super().__init__(problem, budget=problem.n**problem.d, k_init=0, k_samples=1, seed=seed)
        self.points = get_grid(self.problem.d, self.problem.n)
        self.i = 0
    
    def sample_points(self):
        points = self.points[self.i].unsqueeze(0)
        self.i += 1
        return points