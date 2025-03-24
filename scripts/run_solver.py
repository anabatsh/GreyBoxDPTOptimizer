import os
import sys
import json
import torch
import argparse
from tqdm.auto import tqdm

root_path = '../'
sys.path.insert(0, root_path)

from scripts.create_problem import load_problem_set
import solvers as slv


def run_solver(problems, solver, save_dir, budget, n_runs=1):
    solver_class = getattr(slv, solver)
    for problem in tqdm(problems):
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


def main(problem, read_dir, save_dir, suffix, solver, budget, n_runs):
    read_path = os.path.join(read_dir, problem, suffix)
    save_path = os.path.join(save_dir, problem, suffix)
    os.makedirs(save_path, exist_ok=True)

    problems = load_problem_set(read_path)
    run_solver(problems, solver, save_path, budget, n_runs)


if __name__ == '__main__':
    # Set up argparse
    parser = argparse.ArgumentParser(description='Load configuration file.')
    parser.add_argument('--problem', type=str, default="Normal(40, 1)")
    parser.add_argument('--read_dir', type=str, default="../data/normal")
    parser.add_argument('--save_dir', type=str, default="../results/normal_gurobi")
    parser.add_argument('--suffix', type=str, default="train")
    parser.add_argument('--solver', type=str, default="GUROBI")
    parser.add_argument('--n_runs', type=int, default=3)
    parser.add_argument('--budget', type=int, default=5000)

    # Parse arguments
    args = parser.parse_args()
    main(
        args.problem, args.read_dir, args.save_dir, args.suffix, 
        args.solver, args.budget, args.n_runs
    )