import numpy as np
import torch
import os
import json
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from prettytable import PrettyTable
from argparse import ArgumentParser
import problems


def int2bin(x, d, n):
    """
    For a given decimal scalar value obtain a 
    corresponding vector of length d in base-n system
    input: x - batch size decimal scalars in [0, n^d)
           d, n - integer
    input: i - batch size vectors [{1, n}]^d
    """
    i = []
    for _ in range(d):
        i.append(x % n)
        x = x // n
    if isinstance(x, torch.Tensor):
        i = torch.stack(i).T.flip(-1)
    else:
        i = np.array(i)[::-1].T
    return i

def get_xaxis(d, n, len_max=1024):
    """
    For given d and n list all the vectors [{1, n}]^d
    *if there's too many, list only len_max of them with equal step
    input: d, n - integer
    input: i - either all or a part of the vectors [{1, n}]^d
    """
    if d == 1:
        i = np.linspace(0, n-1, len_max).astype(np.int32).reshape(-1, 1)
        return i
    d_new = min(d, int(np.emath.logn(n, len_max)))
    x = np.arange(0, n ** d_new)
    i = int2bin(x, d_new, n)
    if d_new < d:
        i = np.pad(i, ((0, 0), (0, d - d_new)), constant_values=0)
        i = np.pad(i, ((0, 1), (0, 0)), constant_values=n-1)
    return i

def show_problem(problem, save_dir='', ax=None, color='red', linestyle='-o'):
    """
    Vizualize a given problem as a 2D plot and save it. 
    For all (or some reasonable amount) of the points from the problem argument space
    compute the corresponding target values and depict them on a 2D plot, where
    - axis x corresponds to the decimal representation of the argument vectors,
    - axis y corresponds to the target values,
    and the points that don't satisfy the constraints are punctured
    input: problem
           save_dir - directory to save the result
    """
    i = get_xaxis(problem.d, problem.n)
    y = problem.target(i)
    c = problem.constraints(i)
    y[~c] = None

    if ax == None:
        plt.figure(figsize=(8, 4))
        ax = plt.gca()

    ax.set_title('Target Function')
    ax.plot(y, linestyle, c=color, markersize=1)
    min_index = np.argmin(y)
    y_axis = ax.get_ylim()
    ax.vlines(i[min_index], y_axis[0], y[min_index], colors=color)
    ax.set_ylim(y_axis)
    ax.set_xticks([0, len(y)-1], [fr'$[0]^{{{problem.d}}}$', fr'$[{problem.n-1}]^{{{problem.d}}}$'])
    # save_path = os.path.join(save_dir, 'problem.png')
    # plt.savefig(save_path)
    # plt.show()

def print_trajectory(trajectory, problem):
    for _, (state, action, next_state) in enumerate(zip(
            trajectory["states"], 
            trajectory["actions"], 
            trajectory["next_states"]
        )):
        action = int2bin(action, d=problem.d, n=problem.n)
        print(f'step {_} | current state: {state[0].item():>8.6} -> suggested action: {action} -> new target: {next_state[0].item():.6}')
    print()

    query_state = trajectory["query_state"]
    best_found_state = trajectory["next_states"].min(axis=0)
    target_action = trajectory["target_action"]
    target_action = int2bin(target_action, d=problem.d, n=problem.n)
    target_state = problem.target(target_action)

    print(f'query state: {query_state[0].item():.6}')
    print(f'best found state: {best_found_state[0].item():.6}')
    print(f'ground truth state: {target_state[0].item():.6}')
    # print(f'all possible targets in an order:\n{np.sort(all_states)}')

def show_results(read_dir, solvers=[]):
    """
    For given solvers and their reruns
    - summirize the obtained results in a table with the following columns: 
        - the best found target value 
        - the averaged found target value
        - the variance of the found target value
        - the averaged time needed to perform optimization
    - vizualize the optimization processes as best found target value per iteration
    input: read_dir - directory to read the results
           solvers - list of solvers we want to examine
                     if empty, all available solvers will be included
    """
    # accumulate all the optimization processes and results in one dictionary
    solvers = solvers if len(solvers) else os.listdir(read_dir)
    key_list = ('t_best', 'y_best', 'm_list', 'y_list')
    solver_results = {}
    for solver in solvers:
        solver_dir = os.path.join(read_dir, solver)
        if os.path.isdir(solver_dir):
            solver_result = {key: [] for key in key_list}
            for seed in os.listdir(solver_dir):
                seed_dir = os.path.join(solver_dir, seed)
                with open(os.path.join(seed_dir, 'results.json')) as f:
                    r = json.load(f)
                    for key in key_list:
                        solver_result[key].append(r[key])
            # compute average time and target value as well as 
            # the best obtained target value and its variance
            solver_results[solver] = {
                'time': np.mean(solver_result['t_best']),
                'y_best': np.min(solver_result['y_best']),
                'y_mean': np.mean(solver_result['y_best']),
                'y_std': np.std(solver_result['y_best'])
            }
            # compute an average optimization process 
            m_list_full = sum(solver_result['m_list'], [])
            with open(os.path.join(seed_dir, 'results.json')) as f:
                budget = json.load(f)['budget']
            m_min, m_max = np.min(m_list_full), budget #np.max(m_list_full)
            m_intr = np.linspace(m_min, m_max, 10).astype(np.int32)
            y_intr = [
                np.interp(m_intr, m_list, y_list) 
                for (m_list, y_list) in zip(solver_result['m_list'], solver_result['y_list'])
            ]
            solver_results[solver] |= {
                'm_list': m_intr,
                'y_list_mean': np.mean(y_intr, axis=0),
                'y_list_std': np.std(y_intr, axis=0),
            }

    # print the results
    with open(os.path.join(read_dir, 'results.txt'), 'w') as f:
        tb = PrettyTable()
        tb.field_names = ['Solver', 'y best', 'y mean', 'y std', 'time mean']
        tb.add_rows([[
            solver, 
            f'{result["y_best"]: .5f}', 
            f'{result["y_mean"]: .5f}', 
            f'{result["y_std"]: .5f}', 
            f'{result["time"]:.3f}'
        ] for solver, result in solver_results.items()])
        print(tb, file=f)

    # display the vizualization
    plt.figure(figsize=(8, 4))
    plt.title('Optimization Process')
    plt.ylabel('Target Value')
    plt.xlabel('Iteration')

    cmap = cm.get_cmap('jet')
    color_list = [cmap(c) for c in np.linspace(0.1, 0.9, len(solvers))[::-1]]
    for (solver, result), color in zip(solver_results.items(), color_list):
        # averaged optimization process
        plt.plot(result['m_list'], result['y_list_mean'], label=solver, c=color)
        # variance of the process
        plt.fill_between(
            result['m_list'],
            result['y_list_mean']-result['y_list_std'],
            result['y_list_mean']+result['y_list_std'],
            alpha=0.2, color=color
        )
    lgd = plt.legend(facecolor='white', labelcolor='black', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(read_dir, 'results.png'), bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--problem', type=str, required=True, help='Problem')
    parser.add_argument('--d', type=int, default=10, help='Dimension')
    parser.add_argument('--n', type=int, default=2, help='Mode')
    parser.add_argument('--problem_kwargs', type=json.loads, default='{}', help='Additional problem parameters')
    parser.add_argument('--save_dir', type=str, default='', help='Directory to read from and save to')

    args = parser.parse_args()

    # define a problem
    problem_class = getattr(problems, args.problem)
    problem = problem_class(d=args.d, n=args.n, **args.problem_kwargs)

    # save the results to {save dir}/{problem}
    save_dir = os.path.join(args.save_dir, problem.name)
    os.makedirs(save_dir, exist_ok=True)

    # vizualize the problem on 2D plot
    show_problem(problem, save_dir)

    # vizualize the results 
    show_results(save_dir)#, solvers=args.solvers)