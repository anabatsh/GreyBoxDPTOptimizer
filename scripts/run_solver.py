import os
import sys
import json
import argparse
from collections import defaultdict

root_path = '../'
sys.path.insert(0, root_path)

from scripts.create_problem import load_problem_set
import solvers as slv


def run_solver(problems, solver_class, save_path, budget, n_runs=1, full_info=False):
    results = {}
    for problem in problems:
        results[problem.name] = defaultdict(list)

        for seed in range(n_runs):
            solver = solver_class(problem, budget=budget, seed=seed)
            logs = solver.optimize()
            for key in logs.keys() if full_info else ('x_best', 'y_best'):
                results[problem.name][key].append(logs[key])

    with open(f"{save_path}.json", "w") as f:
        json.dump(results, f, indent=4)

def load_results(read_path):
    with open(f"{read_path}.json", "r") as f:
        results = json.load(f)
    return results

def main(problem, read_dir, save_dir, suffix, solver, budget, n_runs, full_info):
    read_path = os.path.join(read_dir, problem, suffix)
    save_dir = os.path.join(save_dir, problem, suffix)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, solver)

    problems = load_problem_set(read_path)
    solver_class = getattr(slv, solver)
    run_solver(problems, solver_class, save_path, budget, n_runs, full_info)

if __name__ == '__main__':
    # Set up argparse
    parser = argparse.ArgumentParser(description='Load configuration file.')
    parser.add_argument('--problem', type=str, default="QUBO__mode_normal__loc_-5__scale_1")
    parser.add_argument('--read_dir', type=str, default="../data/normal")
    parser.add_argument('--save_dir', type=str, default="../results/normal")
    parser.add_argument('--suffix', type=str, default="test")
    parser.add_argument('--solver', type=str, default="PROTES")
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--budget', type=int, default=10)
    parser.add_argument('--full_info', action="store_true", default=True, help="Save full information")

    # Parse arguments
    args = parser.parse_args()
    main(
        args.problem, args.read_dir, args.save_dir, args.suffix, 
        args.solver, args.budget, args.n_runs, args.full_info
    )