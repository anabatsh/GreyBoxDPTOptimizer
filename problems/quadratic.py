import numpy as np
from .base import Problem


class Quadratic(Problem):
    """
    Optimization problem that has a simple quadratic form ax^2 + bx + c = 0
    """
    def __init__(self, d=1, n=1024, seed=0, x_min=-1, x_max=1):
        """
        Additional Input:
            seed - random seed to determine the nets (int)
        """
        super().__init__(d, n)
        np.random.seed(seed)
        self.scaling = (0, n-1)
        self.borders = (x_min, x_max)
        self.coef = np.random.randn(3)
        self.name = f'Quadratic_{seed}'

    def target(self, x):
        x_scaled = (x - self.scaling[0]) / (self.scaling[1] - self.scaling[0])      # [0, n-1] -> [0, 1]
        x_scaled = x_scaled * (self.borders[1] - self.borders[0]) + self.borders[0] # [0, 1] -> [x_min, x_max]
        X_scaled = np.hstack([x_scaled ** 2, x_scaled, np.ones_like(x_scaled)])
        return X_scaled @ self.coef
    
    def constraints(self, x):
        return False