import numpy as np
import os
import json
from time import perf_counter as tpc

LARGE_CONST = 1e+5


def set2text(points, targets, constraints):
    def compress(point): return '(' + ', '.join(str(x) for x in point) + ')'

    if isinstance(targets, float) and isinstance(constraints, bool):
        return f'{compress(points)} -> {targets: .4f} | {constraints}'
    ind = np.argsort(targets) if len(targets) else 0
    return '\n'.join([
        f'{compress(point)} -> {target: .4f} | {constraint}'
        for point, target, constraint in zip(points[ind], targets[ind], constraints[ind])
    ])

class Logger:
    """
    Base class to track sample-update optimization and store the intermediate and final results.
    """
    def __init__(self, solver, save_dir=''):     
        """
        Input:
            solver - solver to track (class, inheriting Solver class)
            save_dir - directory to save results (str)
        """
        self.t_start = tpc()
        self.logs = {
            't_best': None,
            'x_best': None, 
            'y_best': None, 
            'c_best': None,
            'm_list': [],
            'y_list': []
        } | {
            'budget': solver.budget
        }
        self.save_path_logs = os.path.join(save_dir, 'logs.txt')
        self.save_path_results = os.path.join(save_dir, 'results.json')
        self.f = open(self.save_path_logs, 'w')

    def update(self, m, points, targets, constraints, title=''):
        """
        Function to perform updating.
        Input:
            m - current optimization step (int)
            points - points sampled in the step m (iterable of shape [batch_size, d])
            targets - target values corrresponding to the points (iterable of shape [batch_size, d])
            constraints -  constraint flags corrresponding to the points (iterable of shape [batch_size, d])
            title - specific title to label the step (str)
        """
        # if batch_size > 0, get the best point in term of the minimal target value
        i_best = np.argmin(targets)
        x_best = points[i_best]
        y_best = targets[i_best]
        c_best = constraints[i_best]

        print(f'{title} ({m}/{self.logs["budget"]})', file=self.f)
        print(set2text(points, targets, constraints), file=self.f)

        # if the new point is better than the known best point, update the knowledge
        if self.logs['y_best'] is None or (y_best < self.logs['y_best'] and c_best):
            self.logs['t_best'] = tpc() - self.t_start
            self.logs['x_best'] = np.array(x_best, dtype=np.int32).tolist()
            self.logs['y_best'] = float(y_best)
            self.logs['c_best'] = bool(c_best)
            self.logs['m_list'].append(int(m))
            self.logs['y_list'].append(float(y_best))

    def finish(self):
        """
        Function to finish logging.
        """
        print(f'finish', file=self.f)
        print(f'best result ({self.logs["m_list"][-1]}/{self.logs["budget"]})', file=self.f)
        print(set2text(self.logs['x_best'], self.logs['y_best'], self.logs['c_best']), file=self.f)
        self.f.close()

        with open(self.save_path_results, 'w') as f:
            json.dump(self.logs, f, indent=4)

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

    def init_settings(self, seed=0):
        """
        Function to determine the process.
        Input:
            seed - random seed to determine the process (int)
        """
        np.random.seed(seed)

    def init_points(self):
        """
        Function to perform warmstart.
        Output:
            k_init initial points to define a known set D (np.array of shape [k_init, d])
        """
        points = np.random.randint(0, self.problem.n, (self.k_init, self.problem.d))
        return points

    def sample_points(self):
        """
        Function to perform sampling.
        Output:
            k_sample new points to make a sampling step (np.array of shape [k_sample, d])
        """
        points = np.random.randint(0, self.problem.n, (self.k_samples, self.problem.d))
        return points

    def update(self, points, targets, constraints):
        """
        Function to perform updating.
        """
        pass        

    def optimize(self, save_dir=''):
        """
        Function to perform entire optimization.
        Input:
            save_dir - directory to save results (str)
        """
        # define a logger to track the optimization process
        self.logger = Logger(self, save_dir)

        # perform warmastart and generate k_init initial samples of (argument, target, constraints)
        if self.k_init:
            points = self.init_points()
            targets = self.problem.target(points)
            constraints = self.problem.constraints(points)
            self.update(points, targets, constraints)
            self.logger.update(self.k_init, points, targets, constraints, 'warmstarting')

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
                constraints = self.problem.constraints(points)
                self.update(points, targets, constraints)
                self.logger.update(i, points, targets, constraints, 'iteration')
            else:
                # if the algortithm hasn't managed to sample a single point, stop the process
                break
        # close all the files
        self.logger.finish()