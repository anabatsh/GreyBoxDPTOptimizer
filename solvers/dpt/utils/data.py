import glob
import os
import numpy as np
import torch
from functools import partial
from torch.utils.data import Dataset


class MarkovianOfflineDataset(Dataset):
    """
    """
    def __init__(self, problems, seq_len=60, ordered=False, remove_target=False):
        super().__init__()
        self.seq_len = seq_len
        self.remove_target = remove_target
        self.ordered = ordered
        self.problems = problems

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, index: int):
        problem = self.problems[index]
        history_len = problem.n ** problem.d

        x_min = problem.info["x_min"]
        if self.remove_target:
            a = np.hstack([np.arange(history_len)[:x_min], np.arange(history_len)[x_min+1:]])
        else:
            a = np.arange(history_len)
        x = np.random.choice(a, size=self.seq_len, replace=self.seq_len>history_len)
        y = problem.target(x)

        if self.ordered:
            sort_indexes = np.argsort(y)[::-1]
            x = x[sort_indexes]
            y = y[sort_indexes]

        return {
            "x": torch.tensor(x).to(torch.long),
            "y": torch.tensor(y).to(torch.float),
            "x_min": torch.tensor(x_min).to(torch.long),
            "problem": problem
        }

class MarkovianOnlineDataset(Dataset):
    """
    """
    def __init__(self, problems):
        super().__init__()
        self.problems = problems

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, index: int):
        problem = self.problems[index]
        x_min = problem.info["x_min"]
        return {
            "problem": problem,
            "x_min": torch.tensor(x_min).to(torch.long)
        }
