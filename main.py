#! /usr/bin/env python3

from problem import MyNet
import solvers
import torch

from tqdm.auto import tqdm
from argparse import ArgumentParser
import os
import json
import time

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--d', type=int, default=10, help='Problem\'s dimension')
    parser.add_argument('--n', type=int, default=2, help='Problem\'s mode')
    parser.add_argument('--budget', type=int, default=100, help='Budget')
    parser.add_argument('--k_init', type=int, default=10, help='Number of init points')
    parser.add_argument('--k_samples', type=int, default=5, help='Number of sampled points')

    # parser.add_argument('--k_top', type=int, default=100, help='Size of the memory')
    # parser.add_argument('--k_memory', type=int, default=100, help='Size of the memory')

    parser.add_argument('--solver', type=str, help='Solver')
    parser.add_argument('--kwargs', type=json.loads, default='{}', help='Additional parameters')

    parser.add_argument('--save_dir', type=str, default='', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--n_exp', type=int, default=5, help='Number of experiments')
    args = parser.parse_args()


    save_dir = os.path.join(args.save_dir, args.solver)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    problem = MyNet(d=args.d, n=args.n)
    problem.full_plot(save_dir)

    try:
        solver_name = args.solver
        solver_class = getattr(solvers, args.solver)
    except:
        print('Wrong solver')

    solver = solver_class(problem, budget=args.budget, k_init=args.k_init, k_samples=args.k_samples, **args.kwargs)
    for seed in tqdm(range(args.n_exp), desc=solver_name):
        # time.sleep(0.25)
        # for param, val in args.kwargs.items():
        #     print(param, val)
        solver.optimize(seed=seed, save_dir=save_dir)