import nevergrad
import numpy as np
from functools import partial
from .base import Solver

class NgSolver(Solver):
    def __init__(self, problem, budget, k_init=0, k_samples=1, solver=''):
        super().__init__(problem, budget, k_init=0, k_samples=1)

        self.optimizer = solver(
            parametrization=nevergrad.p.TransitionChoice(problem.n, repetitions=problem.d),
            budget=budget,
            num_workers=1,
        )

    def sample_points(self):
        self.points = self.optimizer.ask()
        points = np.array([self.points.value], dtype=int)
        return points
    
    def update(self, points, targets):
        self.optimizer.tell(self.points, targets[0])

PSO = partial(NgSolver, solver=nevergrad.optimizers.PSO)