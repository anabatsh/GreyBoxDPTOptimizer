import numpy as np
from .base import Problem


class Quadratic(Problem):
    """
    Optimization problem that has a simple quadratic form ax^2 + bx + c
    """
    def __init__(self, d=1, n=1024, seed=0):
        """
        Additional Input:
            seed - random seed to determine the nets (int)
        """
        super().__init__(d, n)
        np.random.seed(seed)

        self.coef = np.random.randn(3)
        self.coef[0] = np.abs(self.coef[0]) + 1e-8
        self.base = np.arange(len(self.coef))[::-1]
        self.name = f'Quadratic_{seed}'

    def target(self, i):
        x = i / (self.n ** self.d - 1)
        return np.power(x, self.base) @ self.coef

    def constraints(self, x):
        return np.ones(x.shape[0]).astype(np.bool)

class SimpleQuadratic(Problem):
    """
    Optimization problem that has a simple quadratic form (x - shift)^2
    """
    def __init__(self, d=1, n=1024, seed=0):
        """
        Additional Input:
            seed - random seed to determine the nets (int)
        """
        super().__init__(d, n)
        np.random.seed(seed)

        self.shift = seed / (self.n ** self.d - 1) # np.random.rand(d)
        self.name = f'SimpleQuadratic_{seed}'
        self.params = {"n": n, "d": d, "seed": seed}

        x_max = 0 if self.shift < 0.5 else 1
        y_max = (x_max - self.shift) ** 2
        self.info = {
            "x_min": seed, "y_min": 0,
            "x_max": x_max, "y_max": y_max
        }

    def target(self, i):
        x = i / (self.n ** self.d - 1)
        return ((x - self.shift) ** 2)

    def constraints(self, i):
        return np.ones(i.shape[0]).astype(np.bool)

        # gt = -1 * self.coef[1] / (2 * self.coef[0])
        # self.x_min = gt - (np.random.rand() + 1e-8)
        # self.x_max = gt + (np.random.rand() + 1e-8)
        # self.base = np.arange(len(self.coef))[::-1]
