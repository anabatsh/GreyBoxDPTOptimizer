from os import replace
import numpy as np
import torch
from torch.utils.data import Dataset

def relative_improvement(x, y):
    return np.abs(x - y) / (np.finfo(x.dtype).eps + np.abs(x))


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
        # -----------------------------------------------------------------------------------------
        # тут можно раскомментировать нужный вариант
        # rewards - 1, если достигли минимума или приблизились к нему, иначе 0: [seq_len]
        # rewards = np.zeros(self.seq_len)
        # rewards[mask] = x_next[indices[mask], actions[mask]] == problem.info["x_min"][actions[mask]]
        # rewards[~mask] = np.all(x_next[indices[~mask]] == problem.info["x_min"][None, :], -1)

        # rewards - 1, если достигли минимума, иначе 0: [seq_len]
        # rewards = np.zeros(self.seq_len)
        # rewards[y_next == problem.info["y_min"]] = 1.0

        # R(y) = alpha * relerr(y[n], y[:n].min())) + (1 - alpha) * (1 / (1 + (y[:n] - y[n]).abs().min()))
        alpha = 0.5
        reward_exploit = relative_improvement(np.minimum.accumulate(y), y_next)

        mask = np.tril(np.ones((self.seq_len, self.seq_len), dtype=bool))
        y_expanded = np.where(mask, y[None, :], np.inf)
        exploration = np.abs(y_expanded - y_next[:, None])
        reward_explore = 1 / (1 + exploration.min(1))
        rewards = alpha * reward_exploit + (1 - alpha) * reward_explore

        # query state: [state_dim]

        # до target_action совпадает с минимумом, в target_action отличается, после - произвольный
        # query_x = problem.info["x_min"].copy()
        # if target_action < problem.d:
        #     query_x[target_action] = problem.info["x_min"][target_action] ^ 1
        #     query_x[target_action+1:] = np.random.randint(0, problem.n, size=problem.d-target_action-1)

        # рандомный x "рядом" с argmin - если argmin есть.
        if problem.info["x_min"]:
            query_x = problem.info["x_min"].copy()
            n_corrupt_indices = np.random.binomial(problem.d, 0.4)
            corrupt_indices = np.random.choice(np.arange(problem.d), size=n_corrupt_indices, replace=False) 
            query_x[corrupt_indices] ^= 1
        else:
            # рандомный x
            query_x = np.random.randint(0, problem.n, size=(problem.d))

        y = problem.target(query_x)
        query_state = np.hstack([query_x, y])

        # -----------------------------------------------------------------------------------------
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
            "rewards": torch.FloatTensor(rewards),
            "target_action": torch.tensor(target_action, dtype=torch.long),
            "problem": problem
        }
