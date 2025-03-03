import os
import json
import argparse

import sys
root_path = '../'
sys.path.insert(0, root_path)
import problems as pbm


def save_problem_set(problems, save_path):
    pbm.serialize_problem_set(pbm.ProblemSet(problems), f"{save_path}.dill")

def load_problem_set(read_path):
    return pbm.deserialize_problem_set(f"{read_path}.dill").problems

def create_problem_sets(problem_class, d, n, name, kwargs, n_test, n_val, n_train, save_path):
    problems = [problem_class(d=d, n=n, name=name, seed=i, **kwargs) for i in range(n_test+n_val+n_train)]
    
    problems_test = problems[:n_test]
    problems_val = problems[n_test:n_test+n_val]
    problems_train = problems[n_test+n_val:]

    save_problem_set(problems_test, f"{save_path}/test")
    save_problem_set(problems_val, f"{save_path}/val")
    save_problem_set(problems_train, f"{save_path}/train")

def main(problem, kwargs, d, n, n_train, n_val, n_test, save_dir, name=None):
    if name is None:
        name = problem + f'__n_{n}__d_{d}' + ''.join([f"__{k}_{v}" for k, v in kwargs.items()])
    save_path = os.path.join(save_dir, name)
    os.makedirs(save_path, exist_ok=True)

    problem_class = getattr(pbm, problem)
    create_problem_sets(
        problem_class, d, n, name, kwargs,
        n_test, n_val, n_train, 
        save_path
    )
    return name

if __name__ == '__main__':
    # Set up argparse
    parser = argparse.ArgumentParser(description='Load configuration file.')
    parser.add_argument('--d', type=int, default=50, help='d')
    parser.add_argument('--n', type=int, default=2, help='n')
    parser.add_argument('--n_test', type=int, default=10)
    parser.add_argument('--n_val', type=int, default=10)
    parser.add_argument('--n_train', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default="../data/test")
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--problem', type=str, default="Distribution")
    parser.add_argument('--kwargs', type=json.loads, default={})

    # Parse arguments
    args = parser.parse_args()

    main(
        args.problem, args.kwargs, args.name,
        args.d, args.n, 
        args.n_train, args.n_val, args.n_test, 
        args.save_dir
    )