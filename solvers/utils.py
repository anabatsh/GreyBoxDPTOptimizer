import numpy as np
import matplotlib.pyplot as plt
import os
import json
from matplotlib import colormaps as cm
from prettytable import PrettyTable
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

def show_results(read_dir, solvers=[]):
    key_list = ('time', 'x_best', 'y_best', 'm_list', 'y_list')
    solver_results = {}
    for solver in solvers:#os.listdir(read_dir):
        solver_dir = os.path.join(read_dir, solver)
        if os.path.isdir(solver_dir):
            solver_results[solver] = {key: [] for key in key_list}
            for seed in os.listdir(solver_dir):
                seed_dir = os.path.join(solver_dir, seed)
                if os.path.isdir(seed_dir):
                    with open(os.path.join(seed_dir, 'results.json')) as f:
                        r = json.load(f)
                        for key in key_list:
                            solver_results[solver][key].append(r[key])

    save_path_img = os.path.join(read_dir, 'results.png')
    save_path_txt = os.path.join(read_dir, 'results.txt')
    f = open(save_path_txt, 'w')
    # if os.path.exists(save_path_txt):
    #     os.remove(save_path_txt)

    plt.figure(figsize=(8, 4), facecolor='black')
    ax = plt.axes()
    ax.set_facecolor('black')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(colors='white', which='both')

    plt.title('Optimization Process', color='white')
    plt.ylabel('Target Value')
    plt.xlabel('Iteration')

    cmap = cm.get_cmap('jet')
    c_list = np.linspace(0.1, 0.9, len(solver_results))[::-1]

    tb = PrettyTable()
    tb.field_names = ["Solver", "y best", "y mean", "y std", "time mean"]

    for (solver, results), c in zip(solver_results.items(), c_list):
        color = cmap(c)
        y_best = np.min(results['y_best'])
        y_mean = np.max(results['y_best'])
        y_std = np.std(results['y_best'])
        time_mean = np.mean(results['time'])
        f.write(f'{solver:<10}: best = {y_best: .5f} mean = {y_mean: .5f} std = {y_std: .5f} time = {time_mean:.3f}\n')
        
        tb.add_row([solver, f'{y_best: .5f}', f'{y_mean: .5f}', f'{y_std: .5f}', f'{time_mean:.3f}'])

        m_list_full = sum(results['m_list'], [])
        m_intr = np.linspace(np.min(m_list_full), np.max(m_list_full), 10).astype(np.int32)
        y_intr = [np.interp(m_intr, m_list, y_list) for (m_list, y_list) in zip(results['m_list'], results['y_list'])]

        # for (m_list, y_list) in zip(results['m_list'], results['y_list']):
        #     plt.plot(m_list, y_list, '--', c=color)
        plt.plot(m_intr, np.mean(y_intr, 0), label=solver, c=color)
        plt.fill_between(m_intr, np.min(y_intr, 0), np.max(y_intr, 0), alpha=0.2, color=color)

    lgd = plt.legend(facecolor='black', labelcolor='white', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(save_path_img, bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.show()

    print(tb, file=f)
    f.close()