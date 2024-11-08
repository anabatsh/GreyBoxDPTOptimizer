import random
import numpy as np
import glob
import os
from torch.utils.data import IterableDataset
from typing import List, Dict, Any


def results2trajectories(
        read_dir='results', 
        save_dir='trajectories', 
        solvers=['NoisyBandit', 'PSO', 'OnePlusOne', 'Portfolio', 'PSO', 'SPSA', 'RandomSearch']
    ):

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for problem in os.listdir(read_dir):
        problem_dir = os.path.join(read_dir, problem)
        solvers = solvers if len(solvers) else os.listdir(problem_dir)
        for solver in solvers:
            solver_dir = os.path.join(problem_dir, solver)
            if os.path.isdir(solver_dir):
                for seed in os.listdir(solver_dir):
                    seed_dir = os.path.join(solver_dir, seed)
                    with open(os.path.join(seed_dir, 'logs.txt')) as f:
                        r = f.read().split('\n')[:-4][1::2]
                        arguments = []
                        targets = []
                        constraints = []
                        for row in r:
                            row_argument_target, row_constraint = row.split('|')
                            row_argument, row_target = row_argument_target.split('->')
                            row_argument = list(row_argument.strip()[1:-1].split(','))

                            argument_value = np.array([int(x) for x in row_argument])
                            base_2 = 2 ** np.arange(len(argument_value))[::-1]
                            argument_value = argument_value @ base_2
                            arguments.append(argument_value)

                            target_value = float(row_target)
                            targets.append(target_value)
                            
                            constraint_value = bool(row_constraint)
                            constraints.append(constraint_value)
                            
                    actions = np.array(arguments)
                    states = np.array(targets)
                    ground_truth = actions[np.argmin(states)]

                    n = len(r)
                    history = {
                        "states": np.roll(states, 1),
                        "actions": actions,
                        "target_actions": np.array([ground_truth] * n),
                        "rewards": -1 * (states - np.roll(states, 1))
                    }
                    np.savez(f'{save_dir}/{problem}_{solver}_{seed}', **history, allow_pickle=True)

def load_markovian_learning_histories(path: str) -> List[Dict[str, Any]]:
    files = glob.glob(f"{path}/*.npz")

    learning_histories = []
    for filename in files:
        with np.load(filename, allow_pickle=True) as f:
            learning_histories.append(
                {
                    "states": f["states"],
                    "actions": f["actions"],
                    "target_actions": f["target_actions"],
                    "rewards": f["rewards"]
                }
            )

    return learning_histories

class MarkovianDataset(IterableDataset):
    """
    A dataset class that shuffles the learning histories for dpt training
    a query_state and respective target action for it is sampled from the same
    learning history (to ensure it is related to the same goal with the context) but
    is not related to contextual samples (as in original implementation).

    It is also specified for Dark-Key-To-Door Environment with `has_key` indicator
    to make this env an MDP (because DPT can only solve MDPs, not POMDPs)
    """

    def __init__(self, data_path: str, seq_len: int = 60) -> None:
        super().__init__()
        self.seq_len = seq_len
        print("Loading training histories...")
        self.histories = load_markovian_learning_histories(data_path)
        print(f"Num histories: {len(self.histories)}")

    def __prepare_sample(self, history_idx: int, context_indexes: List[int], query_idx: int):
        history = self.histories[history_idx]

        assert (
            history["states"].shape[0] ==
            history["actions"].shape[0] ==
            history["rewards"].shape[0]
        )

        query_states = history["states"][query_idx].flatten()
        states = history["states"][context_indexes].flatten()
        next_states = history["states"][context_indexes + 1].flatten()
        actions = history["actions"][context_indexes].flatten()
        target_actions = history["target_actions"][query_idx].flatten()
        rewards = history["rewards"][context_indexes].flatten()

        return (
            query_states,
            states,
            actions,
            next_states,
            rewards,
            target_actions,
        )

    def __iter__(self):
        while True:
            history_idx = random.randint(0, len(self.histories) - 1)

            context_indexes = np.random.randint(
                0,
                self.histories[history_idx]["rewards"].shape[0] - 2,
                size=self.seq_len,
            )
            query_idx = random.randint(
                0, self.histories[history_idx]["rewards"].shape[0] - 1
            )

            yield self.__prepare_sample(history_idx, context_indexes, query_idx)