import numpy as np
from .base import Problem


def int2bin(x, d, n):
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
    def __init__(self, d=1, n=2, seed=0):
        """
        Additional Input:
            seed - (int) random seed to determine Q
        """
        super().__init__(d, n)
        np.random.seed(seed)
        self.name += f'_seed_{seed}'

        self.Q = np.triu(np.random.randn(d, d))
        
        x = int2bin(np.arange(0, n ** d), d, n)
        y = self.target(x)
        i_best = np.argmin(y)
        self.info = {"x_min": x[i_best], "y_min": y[i_best]}
        
    def target(self, x):
        return ((x @ self.Q) * x).sum(-1)