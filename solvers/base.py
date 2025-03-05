import torch
from time import perf_counter as tpc

LARGE_CONST = 1e+5


class Logger:
    """
    Base class to track sample-update optimization.
    """
    def __init__(self):
        self.t_start = tpc()
        self.logs = {
            'x_best': None, # best-found argument
            'y_best': None, # best-found target
            't_best': None, # time to find the best solution
            'm_list': [],   # a list of iterations
            'x_list': [],   # a history of best-found arguments
            'y_list': [],   # a history of best-found targets
        }

    def update(self, m, points, targets, constraints):
        """
        Function to perform updating.
        Input:
            m - current optimization step (int)
            points - points sampled in the step m (iterable of shape [batch_size, d])
            targets - target values corrresponding to the points (iterable of shape [batch_size, d])
            constraints -  constraint flags corrresponding to the points (iterable of shape [batch_size, d])
        """
        # if batch_size > 0, get the best point in term of the minimal target value
        i_best = torch.argmin(targets)
        x_best = points[i_best]
        y_best = targets[i_best]

        # if the new point is better than the known best point, update the knowledge
        if self.logs['y_best'] is None or (y_best < self.logs['y_best']):
            self.logs['x_best'] = x_best
            self.logs['y_best'] = y_best
            self.logs['m_list'].append(m)
            self.logs['x_list'].append(x_best)
            self.logs['y_list'].append(y_best)

    def finish(self, save_path=None):
        """
        Function to finish the optimization process.
        """
        self.logs['time'] = tpc() - self.t_start
        self.logs['m_list'] = torch.tensor(self.logs['m_list'])
        self.logs['x_list'] = torch.stack(self.logs['x_list'])
        self.logs['y_list'] = torch.stack(self.logs['y_list'])
        if save_path:
            torch.save(self.logs, f'{save_path}.pt')


class Solver():
    """
    Base class for a sample-update optimizer. 
    """
    def __init__(self, problem, budget, k_init=0, k_samples=1, seed=0):
        """
        Input:
            problem - problem to solve (class, inheriting Problem class)
            budget - maximum number of calls to the target function (int)
            k_init - number of initial points for warmstart (int)
            k_samples - number of points sampled on each step (int)
            seed - random seed to determine the process (int)
        """
        self.problem = problem
        self.budget = budget
        self.k_init = k_init
        self.k_samples = k_samples
        self.init_settings(seed)
        self.name = f"{self.__class__.__name__}"

    def init_settings(self, seed=0):
        """
        Function to determine the process.
        Input:
            seed - random seed to determine the process (int)
        """
        torch.manual_seed(seed)

    def init_points(self):
        """
        Function to perform warmstart.
        Output:
            k_init initial points to define a known set D (np.array of shape [k_init, d])
        """
        points = torch.randint(0, self.problem.n, (self.k_init, self.problem.d))
        return points

    def sample_points(self):
        """
        Function to perform sampling.
        Output:
            k_sample new points to make a sampling step (np.array of shape [k_sample, d])
        """
        points = torch.randint(0, self.problem.n, (self.k_samples, self.problem.d))
        return points

    def update(self, points, targets, constraints):
        """
        Function to perform updating.
        """
        pass        

    def optimize(self, save_path=None):
        """
        Function to perform entire optimization.
        Input:
            save_dir - directory to save results (str)
        """
        # define a logger to track the optimization process
        self.logger = Logger()

        # perform warmastart and generate k_init initial samples of (argument, target, constraints)
        if self.k_init:
            points = self.init_points()
            targets = self.problem.target(points)
            constraints = torch.ones_like(targets) # self.problem.constraints(points)
            self.update(points, targets, constraints)
            self.logger.update(self.k_init, points, targets, constraints)

        # perform sampling and updating until the budget is exhausted # or until convergency
        i = self.k_init
        while i < self.budget:
            # sample new k_sample points
            points = self.sample_points()
            if len(points):
                i += len(points)
                if i > self.budget:
                    # if the total number of the checked points exceeds the budget in this step,
                    # reduce the number of the sampled points in this step 
                    points = points[:self.budget - i]
                    i = self.budget
                targets = self.problem.target(points)
                constraints = torch.ones_like(targets) # self.problem.constraints(points)
                self.update(points, targets, constraints)
                self.logger.update(i, points, targets, constraints)
            else:
                # if the algortithm hasn't managed to sample a single point, stop the process
                break
        self.logger.finish(save_path)
        return self.logger.logs