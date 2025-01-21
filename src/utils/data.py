import numpy as np
import torch
from torch.utils.data import Dataset


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
        return {
            "problem": problem,
            "x_min": torch.tensor(x_min).to(torch.long)
        }

class OfflineDataset(OnlineDataset):
    """
    """
    def __init__(self, problems, seq_len=60, ordered=False, remove_target=False):
        super().__init__(problems)
        self.seq_len = seq_len
        self.remove_target = remove_target
        self.ordered = ordered

    def __getitem__(self, index: int):
        problem = self.problems[index]
        x_min = problem.info["x_min"]

        history_len = problem.n ** problem.d
        x = np.random.choice(history_len, size=self.seq_len, replace=self.seq_len>history_len)
        y = problem.target(x)

        # return {
        #     "x": torch.tensor(x).to(torch.long),
        #     "y": torch.tensor(y).to(torch.float),
        #     "x_min": torch.tensor(x_min).to(torch.long),
        #     "problem": problem
        # }
        return {
            "x": torch.tensor(x / 1023).to(torch.float),
            "y": torch.tensor(y).to(torch.float),
            "x_min": torch.tensor(x_min / 1023).to(torch.float),
            "problem": problem
        }