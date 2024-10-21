#! /bin/bash

PROBLEM_KWARGS='{"seed":1}'
PROBLEM_ARGS=(
    "--problem Net"                      # problem
    "--d 10"                             # dimensionality of the problem
    "--n 2"                              # mode of the problem
    "--problem_kwargs ${PROBLEM_KWARGS}" # additional parameters of the problem
)
RUNNER_ARGS=(
    "--budget 10"        # maximum number of calls to the target function
    "--n_runs 2"         # number of reruns for each solver
    "--save_dir results" # directory to save the results
)

# nevergrad benchmarks
# python ./run.py ${PROBLEM_ARGS[@]} ${RUNNER_ARGS[@]} --solver OnePlusOne
# python ./run.py ${PROBLEM_ARGS[@]} ${RUNNER_ARGS[@]} --solver PSO
# python ./run.py ${PROBLEM_ARGS[@]} ${RUNNER_ARGS[@]} --solver NoisyBandit
# python ./run.py ${PROBLEM_ARGS[@]} ${RUNNER_ARGS[@]} --solver SPSA
# python ./run.py ${PROBLEM_ARGS[@]} ${RUNNER_ARGS[@]} --solver Portfolio

# PROTES benchmark
# python ./run.py ${PROBLEM_ARGS[@]} ${RUNNER_ARGS[@]} --solver PROTES --k_samples 5 --solver_kwargs '{"k_top": 2}'

# vizualize the results
! python ./utils.py ${PROBLEM_ARGS[@]} ${RUNNER_ARGS[2]}

# --------------------- extra benchmarks ---------------------
# bayesian optimization - only for continiuous problems
# python ./run.py ${DEFAULT_ARGS[@]} --solver BO

# the model from "Large Language Models to Enhance Bayesian Optimization" - one needs vpn since openai is used!
# python ./run.py ${DEFAULT_ARGS[@]} --solver LLAMBO --k_samples 5 --kwargs '{"k_memory": 20}'
