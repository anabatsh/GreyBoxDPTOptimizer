import numpy as np
from .base import Problem


class ContinuousQuadratic(Problem):
    """
    Optimization problem that has a simple quadratic form
            alpha * (x - x_shift)^2 + y_shift, x, x_shift in {0, 1}
    """
    def __init__(self, d=1, n=1, seed=0):
        """
        Additional Input:
            seed - random seed to determine the nets (int)
        """
        super().__init__(d, n)
        np.random.seed(seed)
        self.alpha = np.random.lognormal(0, 0.25, d) + 1e-5 # d x LogNorm in [+1e-5, +inf]
        self.x_shift = np.random.randn(d)                   # d x Norm(0, 1) in [-inf, inf] # Uniform in [0, 1]
        self.y_shift = np.random.randn()                    # N(0, 1) in [-inf, +inf]
        
        self.info = {"x_min": self.x_shift, "y_min": self.y_shift}

    def target(self, x):
        return ((x - self.x_shift) ** 2) @ self.alpha + self.y_shift

class DiscreteQuadratic(Problem):
    """
    Optimization problem that has a simple quadratic form
            alpha * (x - x_shift)^2 + y_shift, x in {0, n^d-1}
    """
    def __init__(self, d=1, n=2, seed=0):
        """
        Additional Input:
            seed - random seed to determine the nets (int)
        """
        super().__init__(d, n)
        np.random.seed(seed)

        self.alpha = np.random.lognormal(3, 0.25) + 1e-5 # LogNorm in [+1e-5, +inf]
        self.x_shift = np.random.randint(0, n, size=d)   # d x Integer Uniform in [0, n^d]
        self.y_shift = np.random.randn()                 # N(0, 1) in [-inf, +inf]
        
        self.base = np.pow(n, np.arange(d)[::-1])
        self.scaling = (self.n ** self.d - 1)
        self.info = {"x_min": self.x_shift, "y_min": self.y_shift}

    def target(self, x):
        return self.alpha * ((x - self.x_shift) @ self.base / self.scaling) ** 2 + self.y_shift




    # p = np.random.rand()
    # self.constraint = np.random.choice([0, 1], n**d, p=[p, 1-p]).astype(np.bool)

    # def _constraints(self, x):
    #     return self.constraint[x @ self.base]

    # def target(self, x):
    #     c = self._constraints(x)
    #     y = np.ones_like(c) * 1e+10
    #     if isinstance(c, np.ndarray):
    #         y[c] = self._target(x[c])
    #     elif c:
    #         y = self._target(x)
    #     else:
    #         y = y
    #     return y
