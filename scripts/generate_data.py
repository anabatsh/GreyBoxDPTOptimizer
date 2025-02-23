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


NON_PARALLEL_BASTARDS = ['GUROBI', 'PROTES']


parser = argparse.ArgumentParser(description='Load configuration file.')
parser.add_argument('--d', type=int, default=50, help='d')
parser.add_argument('--n', type=int, default=2, help='n')
parser.add_argument('--n_test', type=int, default=100)
parser.add_argument('--n_val', type=int, default=100)
parser.add_argument('--n_train', type=int, default=2500)
parser.add_argument('--save_data_dir', type=str, default='../data/normal')
parser.add_argument('--save_res_dir', type=str, default='../results/normal')
parser.add_argument('--config', type=str, default='../configs/problem_normal.yaml')
parser.add_argument('--max_workers', type=int, default=10)


if __name__ == '__main__':
    args = parser.parse_args()
    config = load_config(args.config)

    # generate data
    print('Generating data')
    problem_names = process_map(
        partial(
            create_problem, 
            d=args.d, n=args.n, 
            n_test=args.n_test, n_val=args.n_val, n_train=args.n_train, 
            save_dir=args.save_data_dir
        ), 
        [item['problem'] for item in config['problem_list']], 
        [item['kwargs'] for item in config['problem_list']], 
        max_workers=args.max_workers
    )

    # run solvers
    for suffix in ['test', 'val', 'train']:
        print(f'Running {config["solver"]["name"]} on {suffix} set')
        if config['solver']['name'] in NON_PARALLEL_BASTARDS:
            print('[Single-process mode]')
            for problem in tqdm(problem_names):
                run_solver(
                    problem,
                    read_dir=args.save_data_dir, 
                    save_dir=args.save_res_dir, 
                    suffix=suffix, 
                    solver=config['solver']['name'],
                    budget=config['solver']['budget'], 
                    n_runs=config['solver']['n_runs'], 
                    full_info=config['solver']['full_info']
                )
        else:
            process_map(
                partial(
                    run_solver, 
                    read_dir=args.save_data_dir, 
                    save_dir=args.save_res_dir, 
                    suffix=suffix, 
                    solver=config['solver']['name'],
                    budget=config['solver']['budget'], 
                    n_runs=config['solver']['n_runs'], 
                    full_info=config['solver']['full_info']
                ), problem_names, max_workers=args.max_workers
            )

    # run update information
    print('Updating information')
    for suffix in ['test', 'val', 'train']:
        print(f'Updating {suffix} set')
        process_map(
            partial(
                update_info, 
                read_dir=args.save_data_dir,
                save_dir=args.save_res_dir,
                suffix=suffix,
                solver=config['solver']['name']
            ), problem_names, max_workers=args.max_workers
        )