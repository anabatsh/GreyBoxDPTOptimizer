import numpy as np
import torch
from .base import Solver, Logger

import sys
root_path = '../'
sys.path.insert(0, root_path)
from src.train import DPTSolver


class DPT(Solver):
    """
    DPT Solver.
    """
    def __init__(self, problem, budget, k_init=10, k_samples=1, checkpoint='', seed=0):
        super().__init__(problem, budget, 1, 1, seed)
        self.model = DPTSolver.load_from_checkpoint(checkpoint).cpu()

    def optimize(self):
        self.logger = Logger(self)

        query_x = np.random.randint(0, self.problem.n, size=(self.problem.d))
        query_y = self.problem.target(query_x)
        query_state = torch.FloatTensor(np.hstack([query_x, query_y]))
        
        results = self.model.run(
            query_state, self.problem, self.budget, 
            self.model.config["do_sample"], 
            temperature_function=lambda x: self.model.config["temperature"]
        )    
        self.logger.logs['m_list'] = range(1, self.budget + 1)
        self.logger.logs['y_list'] = results["next_states"][:, -1]

        return self.logger.logs