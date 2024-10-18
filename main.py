#! /usr/bin/env python3

import problems
import solvers

import os
import json
from tqdm.auto import tqdm
from argparse import ArgumentParser


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--problem', type=str, required=True, help='Problem')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--d', type=int, default=10, help='Dimension')
    parser.add_argument('--n', type=int, default=2, help='Mode')

    parser.add_argument('--solver', type=str, required=True, help='Solver')
    parser.add_argument('--budget', type=int, default=100, help='Budget')
    parser.add_argument('--k_init', type=int, default=10, help='Number of init points')
    parser.add_argument('--k_samples', type=int, default=5, help='Number of sampled points')
    parser.add_argument('--kwargs', type=json.loads, default='{}', help='Additional parameters')

    parser.add_argument('--n_exp', type=int, default=5, help='Number of experiments')
    parser.add_argument('--save_dir', type=str, default='', help='Directory to save results')
    args = parser.parse_args()

    # create folder {save dir}/{problem}/{problem seed}/{solver}
    problem_dir = args.save_dir # os.path.join(args.save_dir, args.problem)
    solver_dir = os.path.join(problem_dir, args.solver)
    if not os.path.isdir(solver_dir):
        os.makedirs(solver_dir, exist_ok=True)

    # define problem
    problem_class = getattr(problems, args.problem)
    problem = problem_class(d=args.d, n=args.n, seed=args.seed)

    # define solver
    solver_class = getattr(solvers, args.solver)
    solver = solver_class(problem=problem, budget=args.budget, k_init=args.k_init, k_samples=args.k_samples, **args.kwargs)

    # run solver n_exp times and save the results into folders 
    # {save dir}/{problem}/{problem seed}/{solver}/{solver seed}
    for seed in range(args.n_exp): #tqdm
        seed_dir = os.path.join(problem_dir, args.solver)
        if not os.path.isdir(seed_dir):
            os.mkdir(seed_dir, exist_ok=True)
        solver.optimize(seed=seed, save_dir=seed_dir)