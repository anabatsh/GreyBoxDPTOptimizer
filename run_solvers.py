import os
import json
import numpy as np
from tqdm.auto import tqdm


def run(save_dir, problems, solvers, n_runs=10):
    # for every problem in the problem set
    for problem in tqdm(problems, total=len(problems)):

        problem_path = os.path.join(save_dir, problem.name)
        if not os.path.exists(problem_path):
            os.makedirs(problem_path, exist_ok=True)

        # for every solver in the solver list
        for solver_name, solver in solvers:
            solver_path = os.path.join(problem_path, f"{solver_name}.json")

            # solve the problem with the solver for n_runs times
            logs_accumulated = defaultdict(list)

            for seed in range(n_runs):
                logs = solver(problem=problem, seed=seed).optimize()
                for k in ("m_list", "y_list"):
                    logs_accumulated[k].append(logs[k])

            logs_accumulated = {
                "m_list": np.mean(logs_accumulated["m_list"], axis=0).astype(np.int32).tolist(),
                "y_list (mean)": np.mean(logs_accumulated["y_list"], axis=0).tolist(),
                "y_list (std)": np.std(logs_accumulated["y_list"], axis=0).tolist()
            }
            with open(solver_path, 'w') as f:
                json.dump(logs_accumulated, f)