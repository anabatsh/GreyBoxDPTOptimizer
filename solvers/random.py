from .base import Solver


class RandomSearch(Solver):
    def __init__(self, problem, budget, seed=0):
        super().__init__(problem, budget, k_init=0, k_samples=1, seed=seed)