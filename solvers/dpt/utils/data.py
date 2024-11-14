import glob
import os
import numpy as np
import torch
from functools import partial
from torch.utils.data import Dataset


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
                        "states": states.reshape(-1, 1),
                        "actions": actions,
                        "target_actions": np.array([ground_truth] * n)
                    }
                    np.savez(f'{save_dir}/{problem}_{solver}_{seed}', **history, allow_pickle=True)

def load_markovian_learning_histories(path: str):
    learning_histories = []
    files = glob.glob(f"{path}/*.npz")
    for filename in files:
        with np.load(filename, allow_pickle=True) as f:
            learning_histories.append(
                {
                    "states": f["states"],
                    "actions": f["actions"],
                    "target_actions": f["target_actions"],
                    "name": os.path.basename(filename)[:-4],
                }
            )
    return learning_histories

class MarkovianDataset(Dataset):
    """
    A dataset class that shuffles the learning histories for dpt training
    a query_state and respective target action for it is sampled from the same
    learning history (to ensure it is related to the same goal with the context) but
    is not related to contextual samples (as in original implementation).
    """
    def __init__(self, data_path: str, seq_len: int = 60):
        super().__init__()
        self.seq_len = seq_len
        self.histories = load_markovian_learning_histories(data_path)

    def __len__(self):
        return len(self.histories)
    
    def __getitem__(self, index: int):
        history = self.histories[index]
        history_len = history["states"].shape[0]
        assert (
            history["states"].shape[0] ==
            history["actions"].shape[0]
        )
        context_indexes = np.random.randint(1, history_len, size=self.seq_len)
        query_idx = np.random.randint(1, history_len)

        query_state = history["states"][query_idx - 1]
        states = history["states"][context_indexes - 1]
        actions = history["actions"][context_indexes]
        next_states = history["states"][context_indexes]
        rewards = (next_states - states)[:, 0]
        target_action = history["target_actions"][query_idx]
        return {
            "query_state": torch.tensor(query_state).to(torch.float),
            "states": torch.tensor(states).to(torch.float), 
            "actions": torch.tensor(actions).to(torch.long),
            "next_states": torch.tensor(next_states).to(torch.float),
            "rewards": torch.tensor(rewards).to(torch.float),
            "target_action": torch.tensor(target_action).to(torch.long)
        }
        # return {
        #     "query_state": torch.tensor(query_state).to(torch.float),
        #     "actions": torch.tensor(actions).to(torch.long),
        #     "states": torch.tensor(states).to(torch.float), 
        #     "target_action": torch.tensor(target_action).to(torch.long)
        # }
    
class MarkovianOfflineDataset(Dataset):
    """
    """
    def __init__(
            self, 
            problems, 
            query_states,
            transition_function,
            reward_function,
            target_actions,
            seq_len
        ):
        super().__init__()
        self.problems = problems
        self.query_states = query_states
        self.transition_function = transition_function
        self.reward_function = reward_function
        self.target_actions = target_actions
        self.seq_len = seq_len

    def __len__(self):
        return len(self.problems)
    
    def __getitem__(self, index: int):
        return {
            "query_state": torch.tensor(self.query_states[index]).to(torch.float),
            "transition_function": partial(self.transition_function, problem=self.problems[index]), 
            "reward_function": partial(self.reward_function, problem=self.problems[index]),
            "target_action": torch.tensor(self.target_actions[index]).to(torch.long),
            "n_steps": self.seq_len
        }
        # return {
        #     "query_state": torch.tensor(self.query_states[index]).to(torch.float),
        #     "transition_function": partial(self.transition_function, problem=self.problems[index]), 
        #     "target_action": torch.tensor(self.target_actions[index]).to(torch.long),
        #     "n_steps": self.seq_len
        # }