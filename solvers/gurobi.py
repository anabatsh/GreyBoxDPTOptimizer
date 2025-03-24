import torch
import numpy as np

import gurobipy as gp
from gurobipy import GRB

from .base import Logger, Solver


class GUROBI(Solver):
    def __init__(self, problem, budget, seed=0):
        self.seed = seed
        super().__init__(problem, budget, k_init=0, k_samples=1, seed=seed)

    def optimize(self, save_path=None):
        self.logger = Logger()

        intermediate_solutions = []
        intermediate_targets = []

        calls = 0  # to track target calls
        def solutions_callback(model, where):
            nonlocal calls
            if where == GRB.Callback.MIPNODE and calls < self.budget:
                status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
                if status in {GRB.OPTIMAL, GRB.SUBOPTIMAL}:
                    x_val = model.cbGetNodeRel(x)
                    x_array = [int(round(x_val[i])) for i in range(self.problem.d)]
                    xt = torch.LongTensor(x_array)

                    intermediate_solutions.append(xt)
                    intermediate_targets.append(self.problem.target(xt))
                    calls += 1

                    # Early stop once budget is reached
                    if calls >= self.budget:
                        model.terminate()

        model = gp.Model("qubo")
        x = model.addVars(self.problem.d, vtype=GRB.BINARY, name="x")
        obj = gp.quicksum(
            self.problem.Q[i, j].item() * x[i] * x[j]
            for i in range(self.problem.d)
            for j in range(self.problem.d)
        )

        model.setObjective(obj, GRB.MINIMIZE)
        # model._x = x
        model.Params.OutputFlag = 0
        model.setParam(GRB.Param.Seed, self.seed)
        model.setParam(GRB.Param.PoolSearchMode, 1)
        # model.setParam(GRB.Param.PoolSolutions, 5)
        # model.setParam(GRB.Param.PoolGap, 0.5)
        model.setParam(GRB.Param.IterationLimit, 25 * self.budget)

        try:
            model.optimize(solutions_callback)
        except gp.GurobiError:
            pass  # in case it's manually terminated

        x_min = torch.LongTensor([round(x[i].X) for i in range(self.problem.d)])
        y_min = self.problem.target(x_min)

        self.logger.logs['x_best'] = x_min
        self.logger.logs['y_best'] = y_min
        self.logger.logs['x_list'] = intermediate_solutions
        self.logger.logs['y_list'] = intermediate_targets
        self.logger.finish(save_path)
        return self.logger.logs
