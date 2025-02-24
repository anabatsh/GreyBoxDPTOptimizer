import nevergrad
import torch
import numpy as np
from functools import partial
from .base import Solver

class NgSolver(Solver):
    """
    Wrapping Solver for methods from the nevergrad library.
    """
    def __init__(self, problem, solver, budget, seed=0):
        """
        Additional Input:
            solver - name of the particular method
        """
        super().__init__(problem, budget, k_init=0, k_samples=1, seed=seed)

        solver_class = getattr(nevergrad.optimizers, solver)
        self.optimizer = solver_class(
            parametrization=nevergrad.p.TransitionChoice(problem.n, repetitions=problem.d),
            budget=budget,
            num_workers=1,
        )
        
    def init_settings(self, seed=0):
        np.random.seed(seed)
    
    def sample_points(self):
        self.points = self.optimizer.ask()
        points = torch.tensor([self.points.value]).long()
        return points

    def update(self, points, targets, constraints):
        self.optimizer.tell(self.points, targets[0].item())

OnePlusOne = partial(NgSolver, solver='OnePlusOne')
PSO = partial(NgSolver, solver='PSO')
NoisyBandit = partial(NgSolver, solver='NoisyBandit')
SPSA = partial(NgSolver, solver='SPSA')
Portfolio = partial(NgSolver, solver='Portfolio')