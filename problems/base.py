import dill

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

class ProblemSet():
    def __init__(self, problems: list[Problem]):
        self.problems = problems


def serialize_problem_set(problem_set, filename):
    with open(filename, 'wb+') as f:
        dill.dump(problem_set, f)


def deserialize_problem_set(filename):
    with open(filename, 'rb') as f:
        problem_set = dill.load(f)
    return problem_set
