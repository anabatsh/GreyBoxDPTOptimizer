import numpy as np


class GP():
    def __init__(self, d, n):
        self.d = d
        self.n = n
        self.mean = lambda x: 0
        self.covr = lambda x: np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

    def init_set(self, X, Y):
        self.set = [(x, y) for x, y in zip(X, Y)]

    def sample(self, k):
        x = np.random.rand(k, self.d)
        return x

    def update(self, x, y):
        self.set.append((x, y))

    def distribution(self):
        pass