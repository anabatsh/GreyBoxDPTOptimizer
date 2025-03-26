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
from train_dpt import load_config


NON_PARALLEL_BASTARDS = ['GUROBI']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load configuration file.')
    parser.add_argument('--d', type=int, default=50, help='d')
    parser.add_argument('--n', type=int, default=2, help='n')
    parser.add_argument('--n_test', type=int, default=100)
    parser.add_argument('--n_val', type=int, default=100)
    parser.add_argument('--n_train', type=int, default=2500)
    parser.add_argument('--budget', type=int, default=5000)
    parser.add_argument('--n_runs', type=int, default=3)
    parser.add_argument('--save_data_dir', type=str, default='/mnt/data/normal')
    parser.add_argument('--save_res_dir', type=str, default='/mnt/trajectories/normal_gurobi')
    parser.add_argument('--problem_config', type=str, default='../configs/problem_normal.yaml')
    parser.add_argument('--solver_config', type=str, default='../configs/solver.yaml')
    parser.add_argument('--max_workers', type=int, default=16)
    parser.add_argument('--if_problems_exist', type=bool, default=True)
    parser.add_argument('--if_solvers_exist', type=bool, default=True)
    args = parser.parse_args()

    # generate data
    if args.if_problems_exist:
        print('Problems already exist')
        problem_names = os.listdir(args.save_data_dir)
    else:
        print('Creating problems')
        problem_config = load_config(args.problem_config)
        args_list = problem_config['problem_list']

        def wrapper(item):
            return create_problem(
                d=args.d, n=args.n, 
                n_test=args.n_test, n_val=args.n_val, n_train=args.n_train, 
                save_dir=args.save_data_dir,
                **item
            )
        problem_names = process_map(wrapper, args_list, max_workers=args.max_workers)

    # run solvers
    if args.if_solvers_exist:
        print('Solvers already exist')
        solver_config = load_config(args.solver_config)
        solver_names = [item['solver'] for item in solver_config['solver_list']]
    else:
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
                    n_runs=args.n_runs
                )
                if solver in NON_PARALLEL_BASTARDS:
                    print('[Single-process mode]')
                    for problem in tqdm(problem_names):
                        func_run(problem)
                else:
                    process_map(func_run, problem_names, max_workers=args.max_workers)
        solver_names = solvers

    # update info
    print('Updating info')
    for suffix in ['test', 'val', 'train']:
        print(f'Updating {suffix} sets')
        func_info = partial(
            update_info,
            read_data_dir=args.save_data_dir,
            read_res_dir=args.save_res_dir,
            suffix=suffix
        )
        process_map(func_info, problem_names, max_workers=args.max_workers)