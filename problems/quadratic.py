import numpy as np
from .base import Problem


class Quadratic(Problem):
    """
    Optimization problem that has a simple quadratic form
                    alpha * (x - x_shift)^2 + y_shift
    """
    def __init__(self, d=1, n=1024, seed=0):
        """
        Additional Input:
            seed - random seed to determine the nets (int)
        """
        super().__init__(d, n)
        np.random.seed(seed)
        self.alpha = np.exp(np.random.randn()) + 1e-5
        self.x_shift = np.random.randint(0, n)
        self.y_shift = np.random.randn()

        self.scaling = (self.n ** self.d - 1)
        self.info = {"x_min": self.x_shift, "y_min": self.y_shift}

    def target(self, x):
        return self.alpha * ((x - self.x_shift) / self.scaling) ** 2 + self.y_shift
