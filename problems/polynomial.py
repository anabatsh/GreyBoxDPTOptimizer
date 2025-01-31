import numpy as np
from .base import Problem


def generate_alpha(d, x_min):    
    alpha = np.random.uniform(-10, 10, d + 1)
    if d == 1:
        return alpha
    degrees = np.arange(d + 1)[::-1]

    # p''(x_min) > 0
    sd_alpha = alpha[:-2] * degrees[:-2] * degrees[1:-1]
    sd_value = np.pow(x_min, degrees[2:]) @ sd_alpha
    if sd_value <= 0:
        alpha[-3] += (-0.5 * sd_value + 1)

    # p'(x_min) = 0
    fd_alpha = alpha[:-1] * degrees[:-1]
    fd_value = np.pow(x_min, degrees[1:]) @ fd_alpha
    alpha[-2] -= fd_value

    return alpha

class DiscretePolynomial(Problem):
    """
    Optimization problem that has a simple polynomial form
            sum_{n=0}^n alpha_n * x^n,
    where x in {0, n^d-1}
    """
    def __init__(self, d=1, n=2, m=3, seed=0):
        """
        Additional Input:
            seed - random seed to determine the nets (int)
        """
        super().__init__(d, n)
        np.random.seed(seed)
        self.base = np.pow(n, np.arange(d)[::-1])
        self.scaling = (self.n ** self.d - 1)

        x_bin = np.random.randint(0, n, size=d)
        x_dec = (x_bin @ self.base / self.scaling)
        self.alpha = generate_alpha(m, x_dec)
        self.degrees = np.arange(m + 1)[::-1]
        
        x = np.stack([
            np.zeros(d, dtype=x_bin.dtype), 
            x_bin, 
            np.ones(d, dtype=x_bin.dtype) * (n - 1)
        ])
        y = self.target(x)
        i_best = np.argmin(y)
        self.info = {"x_min": x[i_best], "y_min": y[i_best]}

    def target(self, x):
        x = (x @ self.base / self.scaling)
        if isinstance(x, np.ndarray):
            x = x[:, None]
        return np.pow(x, self.degrees) @ self.alpha
