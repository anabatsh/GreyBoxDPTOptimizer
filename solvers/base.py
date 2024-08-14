import numpy as np
import os
import json
from time import perf_counter as tpc

LARGE_CONST = 1e+5

# def set2text(points):
#     def compress(vars): return '(' + ', '.join(str(var) for var in vars) + ')'
#     return '\n'.join([
#         f'{compress(x)} -> {val: .4f}'
#         for x, val in sorted(points.items(), key=lambda x: x[1])
#     ])

def set2text(points, targets):
    def compress(point): return '(' + ', '.join(str(x) for x in point) + ')'
    if isinstance(targets, float):
        return f'{compress(points)} -> {targets: .4f}'
    ind = np.argsort(targets) if len(targets) else 0
    return '\n'.join([
        f'{compress(point)} -> {target: .4f}'
        for point, target in zip(points[ind], targets[ind])
    ])

class Logger:
    def __init__(self):
        self.logs = {
            'time': tpc(), 
            'x_best': None, 
            'y_best': None, 
            'm_list': [], 
            'y_list': []
        }

    def update(self, m, x, y):
        if self.logs['y_best'] is None or y < self.logs['y_best']:
            self.logs['x_best'] = np.array(x, dtype=np.int32).tolist()
            self.logs['y_best'] = float(y)
            self.logs['m_list'].append(int(m))
            self.logs['y_list'].append(float(y))

    def finish(self, budget):
        if self.logs['m_list'][-1] < budget:
            self.logs['m_list'].append(int(budget))
            self.logs['y_list'].append(self.logs['y_best'])
        self.logs['time'] = tpc() - self.logs['time']

class Solver():
    def __init__(self, problem, budget, k_init=0, k_samples=1):
        self.problem = problem
        self.budget = budget
        self.k_init = k_init
        self.k_samples = k_samples

    def init_settings(self, seed=0):
        np.random.seed(seed)

    def init_points(self):
        points = np.random.randint(0, self.problem.n, (self.k_init, self.problem.d))
        targets = self.problem.target(points)
        return points, targets

    def sample_points(self):
        points = np.random.randint(0, self.problem.n, (self.k_samples, self.problem.d))
        return points

    def filter_points(self, points):
        return points

    def update(self, points, targets=None):
        if targets is None:
            targets = self.problem.target(points)
        return points, targets
    
    def optimize(self, seed=0, save_dir=''):
        """
        """
        save_dir = os.path.join(save_dir, str(seed))
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        save_path_logs = os.path.join(save_dir, 'logs.txt')
        f = open(save_path_logs, 'w')
        
        self.init_settings(seed)
        self.logger = Logger()

        if self.k_init:
            points, targets = self.init_points()
            self.update(self.k_init, points, targets)

            i_best = np.argmin(targets)
            self.logger.update(self.k_init, points[i_best], targets[i_best])

            print(f'warmstarting ({self.k_init}/{self.budget})', file=f)
            print(set2text(points, targets), file=f)

        # try:
        i = self.k_init
        while i < self.budget:
            points = self.sample_points()
            points = self.filter_points(points)
            if len(points):
                i += len(points)
                if i > self.budget:
                    points = points[:self.budget - i]
                    i = self.budget

                points, targets = self.update(points)

                i_best = np.argmin(targets)
                self.logger.update(i, points[i_best], targets[i_best])

                print(f'iteration ({i}/{self.budget})', file=f)
                print(set2text(points, targets), file=f)

            else:
                break
        # except:
        #     print('trainig has failed', file=f)

        self.logger.finish(self.budget)
        print(f'finish ({self.logger.logs["m_list"][-1]}/{self.budget})', file=f)

        print(f'best result ({i}/{self.budget})', file=f)
        print(set2text(self.logger.logs['x_best'], self.logger.logs['y_best']), file=f)
        f.close()

        save_path_results = os.path.join(save_dir, 'results.json')
        with open(save_path_results, 'w') as f:
            json.dump(self.logger.logs, f, indent=4)