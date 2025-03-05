import os
import sys
import json
import torch
import argparse
from collections import defaultdict

root_path = '../'
sys.path.insert(0, root_path)

from scripts.create_problem import load_problem_set
import solvers as slv


def run_solver(problems, solver, save_dir, budget, n_runs=1, full_info=False):
    solver_class = getattr(slv, solver)
    for problem in problems:
        for seed in range(n_runs):
            save_path = os.path.join(save_dir, problem.name, solver)
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, f"seed_{seed}")
            solver_class(problem, budget=budget, seed=seed).optimize(save_path)

def load_results(read_path):
    extension = os.path.splitext(read_path)[-1]
    if extension == ".json":
        with open(read_path, "r") as f:
            results = json.load(f)
    elif extension == ".pt":
        results = torch.load(read_path, weights_only=True)
    return results

def main(problem, read_dir, save_dir, suffix, solver, budget, n_runs, full_info):
    read_path = os.path.join(read_dir, problem, suffix)
    save_path = os.path.join(save_dir, problem, suffix)
    os.makedirs(save_path, exist_ok=True)

    problems = load_problem_set(read_path)
    run_solver(problems, solver, save_path, budget, n_runs, full_info)

if __name__ == '__main__':
    # Set up argparse
    parser = argparse.ArgumentParser(description='Load configuration file.')
    parser.add_argument('--problem', type=str, default="QUBO")
    parser.add_argument('--read_dir', type=str, default="../data/test")
    parser.add_argument('--save_dir', type=str, default="../results/test")
    parser.add_argument('--suffix', type=str, default="train")
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