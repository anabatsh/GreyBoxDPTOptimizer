import torch
from .base import Problem
import gurobipy as gp
import numpy as np
from gurobipy import GRB

def int2base(x, d, n):
    """
    Convert a given decimal x to a vector of length d in base-n system
    """
    i = []
    for _ in range(d):
        i.append(x % n)
        x //= n
    return list(reversed(i))

class QUBO(Problem):
    """
    Quadratic Binary Optimization Problem: x^TQx -> min_x
    """
    def __init__(self, d=1, n=2, n_probes=500, seed=0, bound=50, density=0.5,
                 solver="brute", target=None, device='cpu', **kwargs):
        """
        Additional Input:
            seed - (int) random seed to determine Q
        """
        super().__init__(d, n)
        self.device = device
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.d = d
        self.n = n
        self.Q = torch.triu(torch.randn(d, d, generator=self.generator, dtype=torch.float, device=self.device))
        self.name += f"__seed_{seed}"
        self.info = {"x_min": None, "y_min": target}
        self.solver = solver
        self.n_probes = n_probes
        if solver:
            self.find_target()

    def find_target(self):
        if self.solver == 'brute':
            x = torch.tensor([int2base(i, self.d, self.n) for i in range(self.n ** self.d)], device=self.device)
            y = self.target(x)
            i_best = torch.argmin(y).item()
            self.info = {"x_min": x[i_best], "y_min": y[i_best]}
        elif self.solver == 'gurobi':
            intermediate_solutions = []

            def solutions_callback(model, where):
                if where == GRB.Callback.MIPNODE:
                    status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
                    if status in {GRB.OPTIMAL, GRB.SUBOPTIMAL}:
                        x_val = model.cbGetNodeRel(x)
                        x_array = [int(round(x_val[i])) for i in range(self.d)]
                        intermediate_solutions.append(x_array)

            model = gp.Model("qubo")
            x = model.addVars(self.d, vtype=GRB.BINARY, name="x")
            obj = gp.quicksum(self.Q[i, j].item() * x[i] * x[j] for i in range(self.d) for j in range(self.d))
            model.setObjective(obj, GRB.MINIMIZE)
            model._x = x
            model.Params.OutputFlag = 0
            model.setParam(GRB.Param.PoolSearchMode, 1)
            model.setParam(GRB.Param.PoolSolutions, self.n_probes)
            model.setParam(GRB.Param.PoolGap, 0.5)
            model.setParam(GRB.Param.IterationLimit, 50000)
            model.optimize(solutions_callback)

            self.x_probes = torch.tensor(intermediate_solutions, device=self.device, dtype=torch.float)
            if self.x_probes.numel():
                self.y_probes = self.target(self.x_probes)
            x_min = torch.tensor([float(x[i].X) for i in range(self.d)], device=self.device)
            self.info = {"x_min": x_min, "y_min": self.target(x_min)}
        else:
            raise NotImplementedError(f"solver {self.solver} not implemented for {self.__class__.__name__}")

    def target(self, x):
        if not isinstance(self.Q, torch.Tensor):
            self.Q = torch.tensor(self.Q, dtype=torch.float)
        return ((x @ self.Q.to(x.device)) * x).sum(dim=-1).to(x.device)
