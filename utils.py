import re
import os
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
from sklearn.manifold import TSNE

# from matplotlib import pyplot as plt

from scripts.create_problem import load_problem_set
from scripts.run_solver import load_results


# ----------------------test_data.ipynb------------------------------
def get_Xy(read_dir, problem_list, suffix='test'):
    X = []
    y = []
    for i, problem in enumerate(problem_list):
        problems = load_problem_set(f"{read_dir}/{problem}/{suffix}")
        Q = np.stack([problem.Q.numpy() for problem in problems])
        X_i = Q.reshape(Q.shape[0], -1)
        y_i = np.ones(Q.shape[0]) * i
        X.extend(X_i)
        y.extend(y_i)
    X = np.stack(X)
    y = np.stack(y)
    print(X.shape, y.shape)
    return X, y

def get_tsne(X):
    tsne = TSNE(
        n_components=2,      # we want a 2D embedding
        perplexity=30,       # typical perplexity value; can tune
        learning_rate='auto',# can also specify a numeric value like 200
        init='pca',          # good default initialization
        random_state=42      # for reproducibility
    )
    X_tsne = tsne.fit_transform(X)
    return X_tsne

def show_tsne(problem_list, X, y, title=""):
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='datalim')

    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1, len(problem_list)))
    cmap = mcolors.ListedColormap(colors) # hsv

    bounds = np.arange(len(problem_list) + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, norm=norm, s=5)
    colorbar = plt.colorbar(scatter, spacing='proportional')
    tick_positions = (bounds[:-1] + bounds[1:]) / 2
    colorbar.set_ticks(tick_positions)
    colorbar.set_ticklabels(problem_list)

    plt.title(title)
    plt.show()

def print_unique(read_dir, problem_list, suffix='test'):
    for problem_name in problem_list:
        d_x = []
        # d_solver = []
        problems = load_problem_set(f'{read_dir}/{problem_name}/{suffix}')
        for problem in problems:
            d_x.append(problem.info['x_best'])
            # d_solver.append(problem.info['solver'])
        x_unique = torch.unique(torch.stack(d_x), dim=0)
        # solvers, stats = np.unique(d_solver, return_counts=True)
        print(f'{problem_name}: {len(x_unique)}')# {solvers} {stats}')

def get_seed_averaged_results(results):
    d = {'m': [], 'y (mean)': [], 'y (std)': []}
    for problem, logs in results.items():
        m = logs['m_list'][0]
        y = np.minimum.accumulate(logs['y_list'], axis=-1)
        y_mean = np.mean(y, axis=0)
        y_std = np.std(y, axis=0)
        d['m'].append(m)
        d['y (mean)'].append(y_mean)
        d['y (std)'].append(y_std)

    d['m'] = d['m'][0]
    d['y (mean)'] = np.mean(d['y (mean)'], axis=0)
    d['y (std)'] = np.mean(d['y (std)'], axis=0)
    return d

def get_meta_results(read_dir, problem_list, suffix='test'):
    meta_results = {}
    for problem in problem_list:
        meta_results[problem] = {}
        problem_path = os.path.join(read_dir, problem, suffix)
        solver_list = [solver.split('.')[0] for solver in os.listdir(problem_path)]

        for solver in solver_list:
            solver_path = os.path.join(problem_path, solver)
            results = load_results(solver_path)
            seed_averaged_results = get_seed_averaged_results(results)
            meta_results[problem][solver] = seed_averaged_results
    return meta_results

def get_problem_averaged_meta_results(meta_results):
    d = {}
    for problem in meta_results.keys():
        for solver in meta_results[problem].keys():
            if solver not in d:
                d[solver] = {}
            for key in meta_results[problem][solver].keys():
                if key not in d[solver]:
                    d[solver][key] = []
                d[solver][key].append(meta_results[problem][solver][key])
    for solver in d.keys():
        d[solver]['m'] = d[solver]['m'][0]
        d[solver]['y (mean)'] = np.mean(d[solver]['y (mean)'], axis=0)
        d[solver]['y (std)'] = np.mean(d[solver]['y (std)'], axis=0)
    return {'all': d}

def show_meta_results(meta_results):
    problem_list = meta_results.keys()

    m = min(len(problem_list), 4)
    n = int(np.ceil(len(problem_list) / m))

    fig, axes = plt.subplots(n, m, figsize=(4*m, 3*n), gridspec_kw=dict(hspace=0.2, wspace=0.25), sharex=True)
    axes = np.array([axes]) if m == 1 else axes
    axes = axes.reshape(n, m)
    for p, problem in enumerate(problem_list):
        i, j = p // m, p % m

        if 'PROTES' in meta_results[problem]:
            clip_val = meta_results[problem]['PROTES']['y (mean)'][0]
        else:
            clip_val = None

        for solver in meta_results[problem].keys():
            results = meta_results[problem][solver]
            if 'mode' in problem:
                title = re.findall(r'mode_(.*?)(?=__)', problem)[0]
                if title == 'normal':
                    loc = int(re.findall(r'loc_(.*?)(?=__)', problem)[0])
                    scale = int(re.findall(r'scale_(.*)', problem)[0])
                    title = f'N({loc}, {scale})'
            else:
                title = problem
            axes[i, j].set_title(title)
            y = np.clip(results['y (mean)'], None, clip_val)
            axes[i, j].plot(results['m'], y, label=solver)
            # axes[i, j].fill_between(
            #     results['m'], 
            #     y - results['y (std)'], 
            #     y + results['y (std)'], 
            #     alpha=0.3
            # )
        axes[i, j].legend(loc=1)
    plt.show()
# -------------------------------------------------------------------



def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found: {config_path}')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def read_problem(read_path):
    logs_accumulated = {}
    for solver_name in os.listdir(read_path):
        solver_path = os.path.join(read_path, solver_name)
        with open(solver_path, 'r') as f:
            logs = json.load(f)
        logs_accumulated[solver_name[:-5]] = logs
    return logs_accumulated

def read_problemset(read_dir, problems):
    logs_accumulated = defaultdict(list)

    for problem in problems:
        read_path = os.path.join(read_dir, problem.name)
        logs = read_problem(read_path)
        
        for k, v in logs.items():
            logs_accumulated[k].append(v)

    for key, v_list_of_dicts in logs_accumulated.items():
        v_dict_of_lists = defaultdict(list)
        for v_dict in v_list_of_dicts:
            for k, v in v_dict.items():
                v_dict_of_lists[k].append(v)
        logs_accumulated[key] = {k: np.mean(v, axis=0) for k, v in v_dict_of_lists.items()}

    return logs_accumulated

def plot_logs(logs, problems, plot_gt=False, solvers=[], title=""):
    solvers = solvers if len(solvers) else logs.keys()

    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0.05, 0.95, len(solvers)))
    for solver, c in zip(solvers, colors):
        val = logs[solver]
        plt.plot(val['m_list'], val['y_list (mean)'], c=c, label=solver)
        # plt.fill_between(
        #     val['m_list'], 
        #     val['y_list (mean)'] - val['y_list (std)'], 
        #     val['y_list (mean)'] + val['y_list (std)'], 
        #     color=c, alpha=0.2
        # )
    plt.title(f"{title}, {len(problems)} problems")
    _, xmax = plt.gca().get_xlim()
    if plot_gt:
        gt = np.mean([problem.info['y_min'] for problem in problems])
        plt.hlines(gt, 0, xmax, colors='black', linestyle='--', label="Ground Truth", zorder=0)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.8))
    # plt.tight_layout()
    plt.show()