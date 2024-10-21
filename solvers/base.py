import numpy as np
import os
import json
from time import perf_counter as tpc

LARGE_CONST = 1e+5


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
    def __init__(self, save_dir, budget):        
        self.save_dir = save_dir
        save_path_logs = os.path.join(self.save_dir, 'logs.txt')
        self.f = open(save_path_logs, 'w')

        self.budget = budget
        self.logs = {
            'time': tpc(), 
            'x_best': None, 
            'y_best': None, 
            'm_list': [], 
            'y_list': []
        }

    def update(self, m, points, targets, title=''):
        i_best = np.argmin(targets)
        x_best = points[i_best]
        y_best = targets[i_best]

        print(f'{title} ({m}/{self.budget})', file=self.f)
        print(set2text(points, targets), file=self.f)

        if self.logs['y_best'] is None or y_best < self.logs['y_best']:
            self.logs['x_best'] = np.array(x_best, dtype=np.int32).tolist()
            self.logs['y_best'] = float(y_best)
            self.logs['m_list'].append(int(m))
            self.logs['y_list'].append(float(y_best))
        
    def finish(self):
        if self.logs['m_list'][-1] < self.budget:
            self.logs['m_list'].append(int(self.budget))
            self.logs['y_list'].append(self.logs['y_best'])
        self.logs['time'] = tpc() - self.logs['time']

        print(f'finish', file=self.f)
        print(f'best result ({self.logs["m_list"][-1]}/{self.budget})', file=self.f)
        print(set2text(self.logs['x_best'], self.logs['y_best']), file=self.f)

        self.f.close()
        
        save_path_results = os.path.join(self.save_dir, 'results.json')
        with open(save_path_results, 'w') as f:
            json.dump(self.logs, f, indent=4)

class Solver():
    def __init__(self, problem, budget, k_init=0, k_samples=1, seed=0, save_dir=''):
        self.problem = problem
        self.budget = budget
        self.k_init = k_init
        self.k_samples = k_samples

        self.init_settings(seed)
        self.save_dir = save_dir

    def init_settings(self, seed=0):
        np.random.seed(seed)

    def init_points(self):
        points = np.random.randint(0, self.problem.n, (self.k_init, self.problem.d))
        return points

    def sample_points(self):
        points = np.random.randint(0, self.problem.n, (self.k_samples, self.problem.d))
        return points

    def update(self, points, targets):
        pass
        
    def optimize(self):
        # self.init_settings(seed)
        self.logger = Logger(self.save_dir, self.budget)

        if self.k_init:
            points = self.init_points()
            targets = self.problem.target(points)
            self.update(points, targets)
            self.logger.update(self.k_init, points, targets, 'warmstarting')

        # try:
        i = self.k_init
        while i < self.budget:
            points = self.sample_points()
            if len(points):
                i += len(points)
                if i > self.budget:
                    points = points[:self.budget - i]
                    i = self.budget
                targets = self.problem.target(points)
                self.update(points, targets)
                self.logger.update(i, points, targets, 'iteration')
            else:
                break
        # except:
        #     print('trainig has failed', file=f)

        self.logger.finish()