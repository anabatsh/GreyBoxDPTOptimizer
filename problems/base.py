import numpy as np


class Problem():
    """
    Base class for an optimization problem. 
    """
    def __init__(self, d=10, n=2):
        """
        Input:
            d - dimensionality of the problem (int)
            n - mode of the problem (int)
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
        pass