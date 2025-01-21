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
        x_min = problem.info["x_min"]

        if problem.n == 1:
            return {
                "x_min": torch.FloatTensor(x_min),
                "problem": problem
            }
        return {
            "x_min": torch.LongTensor(x_min),
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
        x_min = problem.info["x_min"]

        if problem.n == 1:
            x = np.random.rand(self.seq_len, problem.d)
            y = problem.target(x)
            return {
                "x": torch.FloatTensor(x),
                "y": torch.FloatTensor(y),
                "x_min": torch.FloatTensor(x_min),
                "problem": problem
            }

        x = np.random.randint(0, problem.n, size=(self.seq_len, problem.d))
        y = problem.target(x)
        return {
            "x": torch.LongTensor(x),
            "y": torch.FloatTensor(y),
            "x_min": torch.LongTensor(x_min),
            "problem": problem
        }