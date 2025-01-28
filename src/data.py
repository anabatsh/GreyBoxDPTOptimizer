import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import collate, default_collate_fn_map


def collate_problem_fn(batch, *, collate_fn_map):
    return batch

def custom_collate_fn(batch, problem_class):
    custom_collate_fn_map = default_collate_fn_map | {problem_class: collate_problem_fn}
    return collate(batch, collate_fn_map=custom_collate_fn_map)

class OnlineDataset(Dataset):
    """
    """
    def __init__(self, problems):
        super().__init__()
        self.problems = problems

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, index: int):
        problem = self.problems[index]

        # query state
        x = np.random.randint(0, problem.n, size=(problem.d))
        y = problem.target(x)
        query_state = np.hstack([x, y])

        # # target_action
        # indexes = np.where(problem.info["x_min"] - query_state[:-1].astype(int))[0]
        # index = indexes[0] if len(indexes) else problem.d
        # target_action = np.array([index])

        # target_state
        x = problem.info["x_min"]
        y = problem.info["y_min"]
        target_state = np.hstack([x, y])

        return {
            "query_state": torch.FloatTensor(query_state),
            "target_state": torch.FloatTensor(target_state),
            "problem": problem
        }
    
class OfflineDataset(OnlineDataset):
    """
    """
    def __init__(self, problems, seq_len=60):
        super().__init__(problems)
        self.seq_len = seq_len

    def __getitem__(self, index: int):
        problem = self.problems[index]

        # query state
        x = np.random.randint(0, problem.n, size=(problem.d))
        y = problem.target(x)
        query_state = np.hstack([x, y])

        # states
        x = np.random.randint(0, problem.n, size=(self.seq_len, problem.d))
        y = problem.target(x)
        states = np.hstack([x, y[:, None]])

        # actions
        actions = np.random.randint(0, problem.d + 1, size=(self.seq_len, 1))

        # next states
        x_next = x.copy()
        mask = actions[:, 0] < problem.d
        x_next[np.arange(self.seq_len)[mask], actions[mask][:, 0]] = np.abs(1 - x_next[np.arange(self.seq_len)[mask], actions[mask][:, 0]])        
        y_next = problem.target(x_next)
        next_states = np.hstack([x_next, y_next[:, None]])

        # rewards
        rewards = np.sign(y - y_next)

        # target_action
        indexes = np.where(problem.info["x_min"] - query_state[:-1].astype(int))[0]
        index = indexes[0] if len(indexes) else problem.d
        target_action = np.array([index])

        return {
            "query_state": torch.FloatTensor(query_state),
            "states": torch.FloatTensor(states),
            "actions": torch.LongTensor(actions),
            "next_states": torch.FloatTensor(next_states),
            "rewards": torch.FloatTensor(rewards),
            "target_action": torch.LongTensor(target_action),
            "problem": problem
        }

