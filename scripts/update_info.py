#! /usr/bin/env python3

import os
import sys
import torch
import argparse

root_path = '../'
sys.path.insert(0, root_path)

from scripts.create_problem import load_problem_set, save_problem_set
from scripts.run_solver import load_results


def main(problem, read_data_dir, read_res_dir, suffix):
    read_data_path = os.path.join(read_data_dir, problem, suffix)
    read_res_path = os.path.join(read_res_dir, problem, suffix)
    
    problems = load_problem_set(read_data_path)
    for problem in problems:
        problem_path = os.path.join(read_res_path, problem.name)
        for solver in os.listdir(problem_path):
            solver_path = os.path.join(problem_path, solver)
            for seed in os.listdir(solver_path):
                seed_path = os.path.join(solver_path, seed)
                results = load_results(seed_path)
                suggested_info = {
                    "solver": solver,
                    "x_best": results['x_best'],
                    "y_best": results['y_best']
                }
                if problem.info is None or suggested_info["y_best"] < problem.info["y_best"]:
                    problem.info = suggested_info

    save_problem_set(problems, read_data_path)

if __name__ == '__main__':
    # Set up argparse
    parser = argparse.ArgumentParser(description='Load configuration file.')
    parser.add_argument('--read_data_dir', type=str, default="../data/test")
    parser.add_argument('--read_res_dir', type=str, default="../results/test")
    parser.add_argument('--suffix', type=str, default="test")
    parser.add_argument('--problem', type=str, default="QUBO")

    # Parse arguments
    args = parser.parse_args()
    
    main(args.problem, args.read_data_dir, args.read_res_dir, args.suffix)