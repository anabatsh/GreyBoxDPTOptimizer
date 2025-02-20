import os
import argparse
import numpy as np
from run_solvers import *

import sys
root_path = '../'
sys.path.insert(0, root_path)
import problems as pbm


def get_averaged_results(results):
    averaged_results = {}
    for problem_name, solver_dict in results.items():
        for solver_name, logs in solver_dict.items():
            m_list = logs['m_list'][0]
            y_list = np.mean(np.minimum.accumulate(logs['y_list'], -1), 0)
            y_best = np.mean(logs['y_best'], 0)
            averaged_results[solver_name] = {
                'm_list': m_list, 'y_list': y_list, 'y_best': y_best
            }
    return averaged_results

def get_best_solvers(averaged_results):
    y_best_list = [logs['y_best'] for logs in averaged_results.values()]
    solver_best = list(averaged_results.keys())[np.argmin(y_best_list)]
    return solver_best

if __name__ == '__main__':
    default_problems = [
        "QUBO", "Knapsack", 
        "MaxCut", "WMaxCut", 
        "MVC", "WMVC", 
        "NumberPartitioning"
    ]
    # Set up argparse
    parser = argparse.ArgumentParser(description='Load configuration file.')
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--problem_dir', type=str, default="../data")
    parser.add_argument('--results_dir', type=str, default="../results")
    parser.add_argument('--suffix', type=str, default="test")
    parser.add_argument('--problems', nargs='+', default=default_problems)

    # Parse arguments
    args = parser.parse_args()

    averaged_results = {}
    for problem_name in args.problems:
        read_path = os.path.join(args.results_dir, problem_name, args.suffix)
        results = load_results(read_path)
        averaged_results = get_averaged_results(results)
        solver_best = get_best_solvers(averaged_results)
        main(
            [problem_name], [solver_best], 
            args.problem_dir, args.results_dir, "val", 
            n_runs=args.n_runs, keys=['x_best', 'y_best']
        )
        main(
            [problem_name], [solver_best], 
            args.problem_dir, args.results_dir, "train", 
            n_runs=args.n_runs, keys=['x_best', 'y_best']
        )
