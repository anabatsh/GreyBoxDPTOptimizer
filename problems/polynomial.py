import numpy as np
from .base import Problem


def generate_alpha(d, x_min):    
    alpha = np.random.uniform(-10, 10, d + 1)
    degrees = np.arange(d + 1)[::-1]

    # p'(x_min) = 0
    fd_alpha = alpha[:-1] * degrees[:-1]
    fd_value = np.pow(x_min, degrees[1:]) @ fd_alpha
    alpha[-2:] -= fd_value * np.array([1, -x_min])

    # p''(x_min) > 0
    sd_alpha = alpha[:-2] * degrees[:-2] * degrees[1:-1]
    sd_value = np.pow(x_min, degrees[2:]) @ sd_alpha
    if sd_value <= 0:
        alpha[-3:] += np.array([1, -2 * x_min, x_min ** 2])

    return alpha

class ContinuousPolynomial(Problem):
    """
    Optimization problem that has a simple polynomial form
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
        self.info = {"x_min": x_bin, "y_min": self.target(x_bin)}

    def target(self, x):
        x = (x @ self.base / self.scaling)
        if isinstance(x, np.ndarray):
            x = x[:, None]
        return np.pow(x, self.degrees) @ self.alpha


# import sympy as sp
# def generate_polynomial_with_minimum_1(degree, x_min):
#     x = sp.symbols("x")
#     alpha = np.random.uniform(-10, 10, degree + 1)
#     degrees = np.arange(degree + 1)
#     polynomial = np.pow(x, degrees) @ alpha

#     # p'(x_min) = 0
#     first_derivative = sp.diff(polynomial, x)
#     polynomial -= first_derivative.subs(x, x_min) * (x - x_min)

#     # p''(x_min) > 0
#     second_derivative = sp.diff(first_derivative, x)
#     if second_derivative.subs(x, x_min) <= 0:
#         polynomial += (x - x_min)**2  

#     return polynomial.expand()
