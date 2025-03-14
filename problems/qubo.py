import torch
import numpy as np
import qubogen
import networkx as nx
from torch.distributions.normal import Normal #(mean, std) [0, 1]
from torch.distributions.log_normal import LogNormal #(mean, std) [0, 1]
from torch.distributions.half_normal import HalfNormal # scale 1
from torch.distributions.uniform import Uniform #(low, high) [0, 1]
from torch.distributions.beta import Beta #(concentration1, concentration0) 0.5 0.5
from torch.distributions.cauchy import Cauchy # loc scale [0, 1]
from torch.distributions.gamma import Gamma # concentration rate [1, 1]
from torch.distributions.gumbel import Gumbel # loc, scale 1, 2
from torch.distributions.laplace import Laplace # loc, scale [0, 1]
from torch.distributions.pareto import Pareto # scale, alpha, [1, 1]
from torch.distributions.poisson import Poisson # rate 4
from torch.distributions.von_mises import VonMises # loc, concentration [1, 1]
from .base import Problem


DISTRIBUTIONS = {
    'normal': Normal,
    'log_normal': LogNormal,
    'half_normal': HalfNormal,
    'uniform': Uniform,
    'beta': Beta,
    'cauchy': Cauchy,
    'gamma': Gamma,
    'gumbel': Gumbel,
    'laplace': Laplace,
    'pareto': Pareto,
    'poisson': Poisson,
    'von_mises': VonMises
}

class QUBOBase(Problem):
    """
    Quadratic Binary Optimization Problem: x^TQx -> min_x
    """
    def __init__(self, d=10, n=2, name=None, seed=0, **kwargs):
        """
        Additional Input:
            seed - (int) random seed to determine Q
        """
        super().__init__(d, n)
        self.seed = seed
        self.Q = self.generate_Q()

        if name is None:
            self.name += ''.join([f"__{k}_{v}" for k, v in kwargs.items()])
        else:
            self.name = name
        self.name += f'__seed_{seed}'

    def generate_Q(self):
        pass

    def target(self, x):
        x = x.float()
        Q = self.Q.to(x.device)
        return ((x @ Q) * x).sum(dim=-1)

class Distribution(QUBOBase):
    def __init__(self, d=10, n=2, name=None, seed=0, mode='normal', **kwargs):
        self.mode = mode
        self.distribution = DISTRIBUTIONS[mode](**kwargs)
        super().__init__(d=d, n=n, name=name, seed=seed, mode=mode, **kwargs)

    def generate_Q(self):
        torch.manual_seed(self.seed)
        Q = self.distribution.sample((self.d, self.d)).float()
        Q = 2 * Q - torch.diag(Q.sum(0)) - torch.diag(Q.sum(-1))
        return Q

class QUBO(QUBOBase):
    def generate_Q(self):
        rand = np.random.default_rng(self.seed)
        Q = rand.normal(size=(self.d, self.d))
        Q = np.triu(Q)
        Q = torch.tensor(Q).float()
        return Q

class Knapsack(QUBOBase):
    def generate_Q(self):
        rand = np.random.default_rng(self.seed)
        values = np.diag(rand.random(self.d))
        weights = rand.random(self.d)
        weight_limit = np.mean(weights)
        Q = qubogen.qubo_qkp(value=values, a=weights, b=weight_limit)
        Q = torch.tensor(Q).float()
        return Q

class MaxCut(QUBOBase):
    def generate_Q(self):
        graph = nx.fast_gnp_random_graph(n=self.d, p=0.5, seed=self.seed)
        g = qubogen.Graph.from_networkx(graph)
        Q = qubogen.qubo_max_cut(g=g)
        Q = torch.tensor(Q).float()
        return Q

def qubo_wmax_cut(g, weights):
    n_nodes = g.n_nodes
    q = np.zeros((n_nodes, n_nodes))
    i, j = g.edges.T
    q[i, j] += weights
    q[j, i] += weights
    np.fill_diagonal(q, -q.sum(-1))
    return q

class WMaxCut(QUBOBase):
    def generate_Q(self):
        rand = np.random.default_rng(self.seed)
        graph = nx.fast_gnp_random_graph(n=self.d, p=0.5, seed=self.seed)
        weights = rand.random(len(graph.edges))
        g = qubogen.Graph.from_networkx(graph)
        Q = qubo_wmax_cut(g=g, weights=weights)
        Q = torch.tensor(Q).float()
        return Q
    
class MVC(QUBOBase):
    def generate_Q(self):
        graph = nx.fast_gnp_random_graph(n=self.d, p=0.5, seed=self.seed)
        g = qubogen.Graph.from_networkx(graph)
        Q = qubogen.qubo_mvc(g=g)
        Q = torch.tensor(Q).float()
        return Q

class WMVC(QUBOBase):
    def generate_Q(self):
        rand = np.random.default_rng(self.seed)
        graph = nx.fast_gnp_random_graph(n=self.d, p=0.5, seed=self.seed)
        g = qubogen.Graph.from_networkx(graph)
        g.nodes = rand.random(self.d)
        Q = qubogen.qubo_wmvc(g=g)
        Q = torch.tensor(Q).float()
        return Q
    
class NumberPartitioning(QUBOBase):
    def generate_Q(self):
        rand = np.random.default_rng(self.seed)
        s = rand.integers(0, 10, self.d)
        Q = qubogen.qubo_number_partition(number_set=s)
        Q = torch.tensor(Q).float()
        return Q

class GraphColoring(QUBOBase):
    def generate_Q(self):
        n_color = 3
        assert self.d >= 9, "GraphColoring problem requires d >= 9"
        d = self.d // n_color
        graph = nx.fast_gnp_random_graph(n=d, p=0.5, seed=self.seed)
        g = qubogen.Graph.from_networkx(graph)
        Q = qubogen.qubo_graph_coloring(g, n_color=n_color)
        pad = self.d - d * n_color
        Q = np.pad(Q, ((0, pad), (0, pad)))
        Q = torch.tensor(Q).float()
        return Q

def qubo_qap(flow: np.ndarray, distance: np.ndarray, penalty=10.):
    n = len(flow)
    q = np.einsum("ij,kl->ikjl", flow, distance).astype(np.float32)
    i = range(len(q))
    q[i, :, i, :] += penalty
    q[:, i, :, i] += penalty
    q[i, i, i, i] -= 4 * penalty
    return q.reshape(n ** 2, n ** 2)

class QAP(QUBOBase):
    def generate_Q(self):
        d = int(np.sqrt(self.d))
        rand = np.random.default_rng(self.seed)
        flow = rand.random((d, d))
        flow = (flow + flow.T) / 2
        np.fill_diagonal(flow, 0)
        distance = rand.random((d, d))
        distance = (distance + distance.T) / 2
        np.fill_diagonal(distance, 0)
        Q = qubo_qap(flow=flow, distance=distance)
        pad = self.d - d ** 2
        Q = np.pad(Q, ((0, pad), (0, pad)))
        Q = torch.tensor(Q).float()
        return Q

def qubo_max_clique(g, penalty=10.):
    n_nodes = g.n_nodes
    q = np.zeros((n_nodes, n_nodes))
    i, j = g.edges.T
    q[i, j] += penalty / 2.
    q[j, i] += penalty / 2.
    np.fill_diagonal(q, -1)
    return q

class MaxClique(QUBOBase):
    def generate_Q(self):
        graph = nx.fast_gnp_random_graph(n=self.d, p=0.5, seed=self.seed)
        g = qubogen.Graph.from_networkx(graph)
        Q = qubo_max_clique(g=g)
        Q = torch.tensor(Q).float()
        return Q

def qubo_set_pack(a, weight, penalty=8.):
    assert len(a.shape) == 2
    assert a.shape[-1] == len(weight)
    q = -np.diagflat(weight)
    c = np.einsum("ij,ik->ijk", a, a) / 2.  # constraint
    c *= (1 - np.eye(a.shape[-1]))[None, ...]  # zeros diag
    q += penalty * c.sum(0)
    return q

class SetPack(QUBOBase):
    def generate_Q(self):
        rand = np.random.default_rng(self.seed)
        weights = rand.random(self.d) #rand.integers(1, 5, self.d).astype(np.float32)
        k = rand.integers(1, self.d)
        coefs = rand.integers(0, 2, (self.d, self.d))
        Q = qubo_set_pack(a=coefs, weight=weights)
        Q = torch.tensor(Q).float()
        return Q
    
class SPP(QUBOBase):
    def generate_Q(self):
        rand = np.random.default_rng(self.seed)
        costs = rand.random(self.d) #rand.integers(1, 5, self.d).astype(np.float32)
        k = rand.integers(1, self.d)
        coefs = rand.integers(0, 2, (self.d, self.d))
        Q = qubogen.qubo_spp(cost=costs, set_flag=coefs)
        Q = torch.tensor(Q).float()
        return Q
    
def qubo_max2sat(n, literals, signs):
    q = np.zeros((n, n))
    i, j = literals.T
    si, sj = signs.T
    np.add.at(q, (i[sj], i[sj]), ((-1) ** si)[sj])
    np.add.at(q, (j[si], j[si]), ((-1) ** sj)[si])

    offdiag = np.zeros_like(q)
    np.add.at(offdiag, (i, j), (-1) ** (si ^ sj) / 2.)
    return offdiag + offdiag.T + q

class Max2Sat(QUBOBase):
    def generate_Q(self):
        rand = np.random.default_rng(self.seed)
        k = self.d #rand.integers(1, self.d)
        literals = rand.integers(0, self.d, (k, 2))
        signs = rand.choice([True, False], (k, 2))
        Q = qubo_max2sat(n=self.d, literals=literals, signs=signs)
        Q = torch.tensor(Q).float()
        return Q
    
class Ising(QUBOBase):
    def generate_Q(self):
        rand = np.random.default_rng(self.seed)
        J = np.triu(rand.normal(size=(self.d, self.d)), 1)
        h = rand.normal(size=self.d)
        Q = 2 * J - np.diag(J.sum(0)) - np.diag(J.sum(-1)) - np.diag(h)
        Q = torch.tensor(Q).float()
        return Q