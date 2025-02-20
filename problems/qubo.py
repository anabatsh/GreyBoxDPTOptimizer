import torch
import numpy as np
import qubogen
import networkx as nx
from .base import Problem


class QUBOBase(Problem):
    """
    Quadratic Binary Optimization Problem: x^TQx -> min_x
    """
    def __init__(self, d=1, n=2, seed=0, **kwargs):
        """
        Additional Input:
            seed - (int) random seed to determine Q
        """
        super().__init__(d, n)
        self.seed = seed
        self.name += f"__seed_{seed}"
        self.Q = self.generate_Q()

    def generate_Q(self):
        pass

    def target(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).float()
        Q = self.Q.to(x.device)
        return ((x @ Q) * x).sum(dim=-1)

class QUBO(QUBOBase):
    def generate_Q(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        Q = torch.triu(torch.randn(self.d, self.d, dtype=torch.float, generator=generator))
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
        s = rand.random(self.d)
        Q = qubogen.qubo_number_partition(number_set=s)
        Q = torch.tensor(Q).float()
        return Q
    
class GraphColoring(QUBOBase):
    def generate_Q(self):
        graph = nx.fast_gnp_random_graph(n=self.d, p=0.5, seed=self.seed)
        g = qubogen.Graph.from_networkx(graph)
        Q = qubogen.qubo_graph_coloring(g, n_color=3)
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
        rand = np.random.default_rng(self.seed)
        flow = rand.random((self.d, self.d))
        distance = rand.random((self.d, self.d))
        Q = qubo_qap(flow=flow, distance=distance)
        Q = torch.tensor(Q).float()
        return Q