import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt
from .base import Solver


class BO(Solver):
    def __init__(self, problem, budget, k_init=10, k_samples=1):
        super().__init__(problem, budget, k_init=k_init, k_samples=k_samples)

    def init_settings(self, seed=0):
        np.random.seed(seed)
        self.optimizer = BayesianOptimization(
            f=lambda x: -1 * self.problem.target(x)[0],
            pbounds={'x': (0, self.problem.n)}, 
            random_state=1, 
            verbose=0,
        )

    def init_points(self):
        self.optimizer.maximize(init_points=self.k_init, n_iter=0)
        points = self.optimizer._space.params
        return points

    def sample_points(self):
        self.optimizer.maximize(init_points=0, n_iter=1)
        points = self.optimizer._space.params[-1:]
        return points
    

class BO(Solver):
    """
    Bayesian Optimization Solver.
    """
    def __init__(self, problem, budget, k_init=10, k_samples=1, seed=0):
        super().__init__(problem, budget, k_init, k_samples, seed)

        self.gp = GaussianProcessRegressor(
            kernel=Matern(length_scale_bounds=(1e-6, 1e3), nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=2,
        )
        self.kappa = 1 #2.576
        self.points = []
        self.targets = []

    def init_points(self):
        points = np.random.rand(self.k_init, self.problem.d) * self.problem.n
        return points
    
    def sample_points(self):
        if self.k_init == 0:
            return np.random.rand(self.k_samples, self.problem.d) * self.problem.n
        x = np.tile(np.linspace(0, self.problem.n, 10).reshape(-1, 1), self.problem.d)
        mean, std = self.gp.predict(x, return_std=True)
        y = -1 * (mean + self.kappa * std)
        suggestion = x[np.argmin(y)]
        return np.array([suggestion])

    def update(self, points, targets, constraints):
        self.points.extend(points)
        self.targets.extend(targets)
        self.gp.fit(self.points, self.targets)