import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from create_problems import *

import sys
root_path = '../'
sys.path.insert(0, root_path)
import solvers as slv


def run_solvers(problems, solver_names, save_path, budget, n_runs=1, keys=[]):
        results = {}
        for problem in tqdm(problems):
            results[problem.name] = {}
            for solver_name in solver_names:
                solver_class = getattr(slv, solver_name)
                results[problem.name][solver_name] = defaultdict(list)
                for seed in range(n_runs):
                    solver = solver_class(problem, budget=budget, seed=seed)
                    logs = solver.optimize()
                for key in keys if len(keys) else logs.keys():
                    results[problem.name][solver.name][key].append(logs[key])
        with open(f"{save_path}.json", "w") as f:
            json.dump(results, f, indent=4)

def load_results(read_path):
    with open(f"{read_path}.json", "r") as f:
        results = json.load(f)
    return results

def get_best_results(results):
    best_results = {}
    for problem_name, solver_dict in results.items():
        best_results[problem_name] = defaultdict(list)

        for solver_name, logs in solver_dict.items():
            seed_best = np.argmin(logs['y_best'])
            x_best = logs['x_best'][seed_best]
            y_best = logs['y_best'][seed_best]
            best_results[problem_name]['solver'].append(solver_name)
            best_results[problem_name]['x_best'].append(x_best)
            best_results[problem_name]['y_best'].append(y_best)

        i_best = np.argmin(best_results[problem_name]['y_best'])
        solver_best = best_results[problem_name]['solver'][i_best]
        x_best = best_results[problem_name]['x_best'][i_best]
        y_best = best_results[problem_name]['y_best'][i_best]
        best_results[problem_name] = {'solver': solver_best, 'x_best': x_best, 'y_best': y_best}
    return best_results

def set_info(problems, best_results):
    for problem in problems:
        problem.info = best_results[problem.name]

def main(problem_names, solver_names, read_dir, save_dir, suffix, budget, n_runs, keys=[]):
    for problem_name in problem_names:
        read_path = os.path.join(read_dir, problem_name, suffix)
        problems = load_problem_set(read_path)

        save_path = os.path.join(save_dir, problem_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, suffix)

        run_solvers(problems, solver_names, save_path, budget, n_runs, keys)
        results = load_results(save_path)
        best_results = get_best_results(results)
        set_info(problems, best_results)
        save_problem_set(problems, read_path)

if __name__ == '__main__':
    default_problems = [
        "QUBO", "Knapsack", 
        "MaxCut", "WMaxCut", 
        "MVC", "WMVC", 
        "NumberPartitioning"
    ]
    default_solvers = [
        "RandomSearch"
    ]
    # Set up argparse
    parser = argparse.ArgumentParser(description='Load configuration file.')
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--read_dir', type=str, default="../data")
    parser.add_argument('--save_dir', type=str, default="../results")
    parser.add_argument('--suffix', type=str, default="test")
    parser.add_argument('--budget', type=int, default=1000)
    parser.add_argument('--problems', nargs='+', default=default_problems)
    parser.add_argument('--solvers', nargs='+', default=default_solvers)

    # Parse arguments
    args = parser.parse_args()

    main(args.problems, args.solvers, args.read_dir, args.save_dir, args.suffix, args.budget, args.n_runs)