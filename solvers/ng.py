import nevergrad
import numpy as np
from functools import partial
from .base import Solver

class NgSolver(Solver):
    """
    Wrapping Solver for methods from the nevergrad library.
    """
    def __init__(self, problem, budget, k_init=0, k_samples=1, solver='', seed=0):
        """
        Additional Input:
            solver - name of the particular method
        """
        super().__init__(problem, budget, 0, 1, seed)

        self.optimizer = solver(
            parametrization=nevergrad.p.TransitionChoice(problem.n, repetitions=problem.d),
            budget=budget,
            num_workers=1,
        )

    def sample_points(self):
        self.points = self.optimizer.ask()
        points = np.array([self.points.value], dtype=int)
        return points
    
    def update(self, points, targets, constraints):
        self.optimizer.tell(self.points, targets[0])

OnePlusOne = partial(NgSolver, solver=nevergrad.optimizers.OnePlusOne)
PSO = partial(NgSolver, solver=nevergrad.optimizers.PSO)
NoisyBandit = partial(NgSolver, solver=nevergrad.optimizers.NoisyBandit)
SPSA = partial(NgSolver, solver=nevergrad.optimizers.SPSA)
Portfolio = partial(NgSolver, solver=nevergrad.optimizers.Portfolio)