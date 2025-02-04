from os import replace
import numpy as np
import torch
from torch.utils.data import Dataset


class OnlineDataset(Dataset):
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

        # # target_state
        # x = problem.info["x_min"]
        # y = problem.info["y_min"]
        # target_state = np.hstack([x, y])

        return {
            "query_state": torch.FloatTensor(query_state),
            # "target_state": torch.FloatTensor(target_state),
            "problem": problem
        }

class OfflineDataset(OnlineDataset):
    def __init__(self, problems, seq_len=50):
        super().__init__(problems)
        self.seq_len = seq_len

    def __getitem__(self, index: int):
        problem = self.problems[index]

        # states: [seq_len, state_dim]
        x = np.random.randint(0, problem.n, size=(self.seq_len, problem.d))
        y = problem.target(x)
        states = np.hstack([x, y[:, None]])

        # actions: [seq_len]
        actions = np.random.randint(0, problem.d + 1, size=self.seq_len)

        # next states: [seq_len, state_dim]
        x_next = x.copy()
        mask = actions < problem.d
        indices = np.arange(self.seq_len)
        x_next[indices[mask], actions[mask]] ^= 1
        y_next = problem.target(x_next)
        next_states = np.hstack([x_next, y_next[:, None]])

        # query state: [state_dim]
        query_x = np.random.randint(0, problem.n, size=(problem.d))
        query_y = problem.target(query_x)
        query_state = np.hstack([query_x, query_y])

        # target_action: []
        # максимизирующий exploitation
        possible_actions = np.eye(problem.d + 1, problem.d, dtype=int)
        possible_next_states = possible_actions ^ query_x
        target_action = problem.target(possible_next_states).argmin()

        return {
            "query_state": torch.FloatTensor(query_state),
            "states": torch.FloatTensor(states),
            "actions": torch.LongTensor(actions),
            "next_states": torch.FloatTensor(next_states),
            "target_action": torch.tensor(target_action, dtype=torch.long),
            "problem": problem
        }
