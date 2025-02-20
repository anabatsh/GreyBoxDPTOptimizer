import os
import argparse

import sys
root_path = '../'
sys.path.insert(0, root_path)
import problems as pbm


def save_problem_set(problems, save_path):
    pbm.serialize_problem_set(pbm.ProblemSet(problems), f"{save_path}.dill")

def load_problem_set(read_path):
    return pbm.deserialize_problem_set(f"{read_path}.dill").problems

def create_problem_sets(problem_name, d, n, n_test, n_val, n_train, save_path):
    problem_class = getattr(pbm, problem_name)
    problems = [problem_class(d=d, n=n, seed=i) for i in range(n_test+n_val+n_train)]
    
    problems_test = problems[:n_test]
    problems_val = problems[n_test:n_test+n_val]
    problems_train = problems[n_test+n_val:]

    save_problem_set(problems_test, f"{save_path}/test")
    save_problem_set(problems_val, f"{save_path}/val")
    save_problem_set(problems_train, f"{save_path}/train")

def load_problem_sets(read_dir):
    problems_test = {}
    problems_val = {}
    problems_train = {}

    for problem_name in os.listdir(read_dir):
        read_path = os.path.join(read_dir, problem_name)
        problems_test[problem_name] = load_problem_set(f"{read_path}/test")
        problems_val[problem_name] = load_problem_set(f"{read_path}/val")
        problems_train[problem_name] = load_problem_set(f"{read_path}/train")
    return problems_test, problems_val, problems_train

if __name__ == '__main__':
    default_problems = [
        "QUBO", "Knapsack", 
        "MaxCut", "WMaxCut", 
        "MVC", "WMVC", 
        "NumberPartitioning"
    ]
    # Set up argparse
    parser = argparse.ArgumentParser(description='Load configuration file.')
    parser.add_argument('--d', type=int, default=5, help='d')
    parser.add_argument('--n', type=int, default=2, help='n')
    parser.add_argument('--n_test', type=int, default=10)
    parser.add_argument('--n_val', type=int, default=10)
    parser.add_argument('--n_train', type=int, default=50)
    parser.add_argument('--save_dir', type=str, default="../data")
    parser.add_argument('--problems', nargs='+', default=default_problems)

    # Parse arguments
    args = parser.parse_args()

    for problem_name in args.problems:
        save_path = os.path.join(args.save_dir, problem_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        create_problem_sets(
            problem_name, args.d, args.n, 
            args.n_test, args.n_val, args.n_train, 
            save_path
        )