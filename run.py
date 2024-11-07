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
    parser.add_argument('--d', type=int, default=10, help='Dimensionality of the problem')
    parser.add_argument('--n', type=int, default=2, help='Mode of the problem')
    parser.add_argument('--problem_kwargs', type=json.loads, default='{}', help='Additional parameters of the problem')

    parser.add_argument('--solver', type=str, required=True, help='Solver')
    parser.add_argument('--budget', type=int, default=100, help='Budget')
    parser.add_argument('--k_init', type=int, default=10, help='Number of initial points for warmstart')
    parser.add_argument('--k_samples', type=int, default=5, help='Number of points sampled on each step')
    parser.add_argument('--solver_kwargs', type=json.loads, default='{}', help='Additional parameters of the solver')

    parser.add_argument('--n_runs', type=int, default=5, help='Number of the solver reruns (different seeds are used)')
    parser.add_argument('--save_dir', type=str, default='', help='Directory to save results')
    args = parser.parse_args()

    # define a problem
    problem_class = getattr(problems, args.problem)
    problem = problem_class(d=args.d, n=args.n, **args.problem_kwargs)

    # define a solver
    solver_class = getattr(solvers, args.solver)

    # run the solver for n_runs times
    for seed in tqdm(range(args.n_runs), desc=args.solver):
        # save the results to {save dir}/{problem}/{solver}/{solver seed}
        save_dir = os.path.join(args.save_dir, problem.name, args.solver, str(seed))
        os.makedirs(save_dir, exist_ok=True)
        solver = solver_class(
            problem=problem, budget=args.budget, 
            k_init=args.k_init, k_samples=args.k_samples,
            seed=seed,
            **args.solver_kwargs
        )
        solver.optimize(save_dir=save_dir)