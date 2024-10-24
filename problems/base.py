import numpy as np


class Problem():
    """
    Base class for an optimization problem. 
    """
    def __init__(self, d=10, n=2):
        """
        Input:
            d - number of points sampled on each step (int)
            n - random seed to determine the process (int)
        """
        self.d = d
        self.n = n

    def target(self, x):
        """
        Function to compute target values corresponding to given arguments x.
        Input:
            x - given arguments (integer vectors of shape [batsh_size, d])
        Output:
            y - target values (float vector of shape [batch_size])
        """
        return np.random.rand(*np.array(x).shape)
    
    def constraints(self, x):
        """
        Function to compute whether given argument x satisfy the constraints.
        Input:
            x - given arguments (integer vectors of shape [batsh_size, d])
        Output:
            y - constraints compliance (boolean vector of shape [batch_size])
        """
        return np.random.randint(0, 2, *np.array(x).shape).astype(np.bool_)