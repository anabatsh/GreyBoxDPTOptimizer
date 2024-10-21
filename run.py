#! /usr/bin/env python3

import os
import json
from tqdm.auto import tqdm
from argparse import ArgumentParser

import problems
import solvers


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--problem', type=str, required=True, help='Problem')
    parser.add_argument('--d', type=int, default=10, help='Dimension')
    parser.add_argument('--n', type=int, default=2, help='Mode')
    parser.add_argument('--problem_kwargs', type=json.loads, default='{}', help='Additional problem parameters')

    parser.add_argument('--solver', type=str, required=True, help='Solver')
    parser.add_argument('--budget', type=int, default=100, help='Budget')
    parser.add_argument('--k_init', type=int, default=10, help='Number of init points')
    parser.add_argument('--k_samples', type=int, default=5, help='Number of sampled points')
    parser.add_argument('--solver_kwargs', type=json.loads, default='{}', help='Additional solver parameters')

    parser.add_argument('--n_exp', type=int, default=5, help='Number of experiments')
    parser.add_argument('--save_dir', type=str, default='', help='Directory to save results')
    args = parser.parse_args()

    # define problem
    problem_class = getattr(problems, args.problem)
    problem = problem_class(d=args.d, n=args.n, **args.problem_kwargs)

    # define solver
    solver_class = getattr(solvers, args.solver)

    # run solver n_exp times and save the results into folders 
    # {save dir}/{problem}/{solver}/{solver seed}
    for seed in tqdm(range(args.n_exp), desc=args.solver):
        save_dir = os.path.join(args.save_dir, args.problem, args.solver, str(seed))
        os.makedirs(save_dir, exist_ok=True)
        solver = solver_class(
            problem=problem, budget=args.budget, 
            k_init=args.k_init, k_samples=args.k_samples,
            seed=seed, save_dir=save_dir,
            **args.solver_kwargs
        )
        solver.optimize()