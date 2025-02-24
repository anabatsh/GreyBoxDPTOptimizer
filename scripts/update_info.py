#! /usr/bin/env python3

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict

root_path = '../'
sys.path.insert(0, root_path)

from scripts.create_problem import load_problem_set, save_problem_set
from scripts.run_solver import load_results


def set_info(problems, results, solver):
    for problem in problems:
        logs = results[problem.name]
        seed_best = np.argmin(logs["y_best"])
        suggested_info = {
            "solver": solver, 
            "x_best": logs['x_best'][seed_best], 
            "y_best": logs['y_best'][seed_best]
        }
        if problem.info is None or suggested_info["y_best"] < problem.info["y_best"]:
            problem.info = suggested_info

def main(problem, read_dir, save_dir, suffix, solver):
    read_data_path = os.path.join(read_dir, problem, suffix)
    read_res_path = os.path.join(save_dir, problem, suffix, solver)
    
    problems = load_problem_set(read_data_path)
    results = load_results(read_res_path)
    set_info(problems, results, solver)
    save_problem_set(problems, read_data_path)

if __name__ == '__main__':
    # Set up argparse
    parser = argparse.ArgumentParser(description='Load configuration file.')
    parser.add_argument('--read_dir', type=str, default="../data")
    parser.add_argument('--save_dir', type=str, default="../results")
    parser.add_argument('--suffix', type=str, default="test")
    parser.add_argument('--problem', type=str, default="QUBO")
    parser.add_argument('--solver', type=str, default="RandomSearch")

    # Parse arguments
    args = parser.parse_args()
    
    main(args.problem, args.read_dir, args.save_dir, args.suffix, args.solver)