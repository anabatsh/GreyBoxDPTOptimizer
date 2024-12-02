import glob
import os
import numpy as np
import torch
from functools import partial
from torch.utils.data import Dataset


def load_markovian_learning_histories(path: str):
    learning_histories = []
    files = glob.glob(f"{path}/*.npz")
    for filename in files:
        with np.load(filename, allow_pickle=True) as f:
            learning_histories.append(
                {
                    "states": f["states"],
                    "actions": f["actions"],
                    "rewards": f["rewards"],
                    "target_actions": f["target_actions"]
                }
            )
    return learning_histories

class MarkovianOfflineDataset(Dataset):
    """
    A dataset class that shuffles the learning histories for dpt training
    a query_state and respective target action for it is sampled from the same
    learning history (to ensure it is related to the same goal with the context) but
    is not related to contextual samples (as in original implementation).
    """
    def __init__(self, data_path: str, seq_len: int = 60, ordered: bool = False):
        super().__init__()
        self.seq_len = seq_len
        self.ordered = ordered
        self.histories = load_markovian_learning_histories(data_path)

    def __len__(self):
        return len(self.histories)
    
    def __getitem__(self, index: int):
        history = self.histories[index]
        history_len = history["states"].shape[0] - 1
        assert (
            history_len ==
            history["actions"].shape[0] == 
            history["rewards"].shape[0] ==
            history["target_actions"].shape[0]
        )
        context_indexes = np.random.choice(history_len, size=self.seq_len, replace=self.seq_len>history_len)
        query_idx = np.random.choice(history_len)
        context_indexes[0] = history["target_actions"][0]
        context_indexes[10] = history["target_actions"][0]
        context_indexes[20] = history["target_actions"][0]
        context_indexes[30] = history["target_actions"][0]
        if self.ordered:
            sort_indexes = np.argsort(history["states"][context_indexes + 1], axis=0)[::-1]
            context_indexes = context_indexes[sort_indexes]
        return {
            "query_state": torch.tensor(history["states"][query_idx]).to(torch.float),
            "states": torch.tensor(history["states"][context_indexes]).to(torch.float), 
            "actions": torch.tensor(history["actions"][context_indexes]).to(torch.long),
            "next_states": torch.tensor(history["states"][context_indexes + 1]).to(torch.float),
            "rewards": torch.tensor(history["rewards"][context_indexes]).to(torch.float),
            "target_action": torch.tensor(history["target_actions"][query_idx]).to(torch.long)
        }

class MarkovianOnlineDataset(Dataset):
    """
    """
    def __init__(
            self, 
            problems, 
            query_states,
            transition_function,
            reward_function,
            target_actions
        ):
        super().__init__()
        self.problems = problems
        self.query_states = query_states
        self.transition_function = transition_function
        self.reward_function = reward_function
        self.target_actions = target_actions

    def __len__(self):
        return len(self.problems)
    
    def __getitem__(self, index: int):
        return {
            "query_state": torch.tensor(self.query_states[index]).to(torch.float),
            "transition_function": partial(self.transition_function, problem=self.problems[index]), 
            "reward_function": partial(self.reward_function, problem=self.problems[index]),
            "target_action": torch.tensor(self.target_actions[index]).to(torch.long)
        }