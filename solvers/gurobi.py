import gurobipy as gp
import numpy as np
from gurobipy import GRB
import torch
from .base import Logger, Solver


class GUROBI(Solver):
    def __init__(self, problem, budget, k_init=0, k_samples=0, seed=0):
        super().__init__(problem, budget, k_init, k_samples, seed)

    def optimize(self):
        self.logger = Logger(self)

        device = "cpu"
        intermediate_solutions = []

        def solutions_callback(model, where):
            if where == GRB.Callback.MIPNODE:
                status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
                if status in {GRB.OPTIMAL, GRB.SUBOPTIMAL}:
                    x_val = model.cbGetNodeRel(x)
                    x_array = [int(round(x_val[i])) for i in range(self.problem.d)]
                    intermediate_solutions.append(x_array)

        model = gp.Model("qubo")
        x = model.addVars(self.problem.d, vtype=GRB.BINARY, name="x")
        obj = gp.quicksum(self.problem.Q[i, j].item() * x[i] * x[j] for i in range(self.problem.d) for j in range(self.problem.d))
        model.setObjective(obj, GRB.MINIMIZE)
        model._x = x
        model.Params.OutputFlag = 0
        model.setParam(GRB.Param.PoolSearchMode, 1)
        model.setParam(GRB.Param.PoolSolutions, self.budget)
        model.setParam(GRB.Param.PoolGap, 0.5)
        model.setParam(GRB.Param.IterationLimit, 50000)
        model.optimize(solutions_callback)

        x_probes = torch.tensor(intermediate_solutions, device=device, dtype=torch.float)
        if x_probes.numel():
            y_probes = self.problem.target(x_probes)
        x_min = torch.tensor([float(x[i].X) for i in range(self.problem.d)], device=device)
        y_min = self.problem.target(x_min)

        x_min = x_min.numpy()
        y_min = y_min.numpy()
        self.logger.logs['x_best'] = x_min.tolist()
        self.logger.logs['y_best'] = float(y_min)
        self.logger.logs['y_list'] = [float(y_min)]
        self.logger.logs['m_list'] = [self.budget]
        return self.logger.logs