#! /usr/bin/env python3

import os
import sys
import argparse
from functools import partial
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

root_path = '../'
sys.path.insert(0, root_path)
from scripts.create_problem import main as create_problem
from scripts.run_solver import main as run_solver
from scripts.update_info import main as update_info
from utils import load_config


NON_PARALLEL_BASTARDS = ['GUROBI']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load configuration file.')
    parser.add_argument('--d', type=int, default=50, help='d')
    parser.add_argument('--n', type=int, default=2, help='n')
    parser.add_argument('--n_test', type=int, default=2)
    parser.add_argument('--n_val', type=int, default=2)
    parser.add_argument('--n_train', type=int, default=5)
    parser.add_argument('--budget', type=int, default=100)
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--save_data_dir', type=str, default='../data/normal')
    parser.add_argument('--save_res_dir', type=str, default='../results/normal')
    parser.add_argument('--problem_config', type=str, default='../configs/problem_normal.yaml')
    parser.add_argument('--solver_config', type=str, default='../configs/solver.yaml')
    parser.add_argument('--max_workers', type=int, default=10)

    args = parser.parse_args()

    # generate data
    print('Creating problems')
    problem_config = load_config(args.problem_config)
    problems = [item['problem'] for item in problem_config['problem_list']]
    problem_kwargs = [item['kwargs'] for item in problem_config['problem_list']]
    func_data = partial(
        create_problem, 
        d=args.d, n=args.n, 
        n_test=args.n_test, n_val=args.n_val, n_train=args.n_train, 
        save_dir=args.save_data_dir
    )
    problem_names = process_map(func_data, problems, problem_kwargs, max_workers=args.max_workers)

    # run solvers
    print('Running solvers')
    solver_config = load_config(args.solver_config)
    solvers = [item['solver'] for item in solver_config['solver_list']]
    solver_kwargs = [item['kwargs'] for item in solver_config['solver_list']]
    for suffix in ['test', 'val', 'train']:
        for solver, kwargs in zip(solvers, solver_kwargs):
            print(f'Running {solver} on {suffix} sets')
            func_run = partial(
                run_solver,
                read_dir=args.save_data_dir, 
                save_dir=args.save_res_dir, 
                suffix=suffix,
                solver=solver,
                budget=args.budget,
                n_runs=args.n_runs,
                full_info=True
            )
            if solver in NON_PARALLEL_BASTARDS:
                print('[Single-process mode]')
                for problem in tqdm(problem_names):
                    func_run(problem)
            else:
                process_map(func_run, problem_names, max_workers=args.max_workers)

    # update info
    print('Updating info')
    for suffix in ['test', 'val', 'train']:
        for solver in solvers:
            print(f'Updating {suffix} sets with {solver}')
            func_info = partial(
                update_info, 
                read_dir=args.save_data_dir,
                save_dir=args.save_res_dir,
                suffix=suffix,
                solver=solver
            )
            process_map(func_info, problem_names, max_workers=args.max_workers)