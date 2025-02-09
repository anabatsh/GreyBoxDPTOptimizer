import os
import json
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt


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

def plot_logs(logs, problems, solvers=[]):
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
    plt.title(f"{len(problems)} problems")
    gt = np.mean([problem.info['y_min'] for problem in problems])
    _, xmax = plt.gca().get_xlim()
    plt.hlines(gt, 0, xmax, colors='black', linestyle='--', label="Ground Truth", zorder=0)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.8))
    plt.show()