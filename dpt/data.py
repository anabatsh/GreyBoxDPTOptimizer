import torch
from torch.utils.data import Dataset
import numpy as np

class OnlineDataset(Dataset):
    def __init__(self, problems, device='cpu'):
        super().__init__()
        self.problems = problems
        self.device = device

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, index: int):
        problem = self.problems[index]

        # query state
        x = torch.randint(0, problem.n, (problem.d,), dtype=torch.float, device=self.device)
        y = problem.target(x)
        query_state = torch.cat([x, y.unsqueeze(0)])

        # target_state
        if problem.info is None:
            target_state = query_state
        else:
            x = problem.info["x_best"]
            y = problem.info["y_best"]
            target_state = torch.cat([x, y.unsqueeze(0)])

        # if problem.info["y_min"] is None and problem.solver is not None:
        #     problem.find_target()
        # if problem.info["y_min"] is not None:
        #     x = problem.info["x_min"]
        #     y = problem.info["y_min"]
        #     target_state = torch.cat([x, y.unsqueeze(0)])
        # else:
        #     target_state = query_state

        return {
            "query_state": query_state.float(),
            "target_state": target_state.float(),
            "problem": problem,
        }


class OfflineDataset(OnlineDataset):
    def __init__(self, problems, seq_len=50, ad_eps=(0.1, 0.9), device='cpu'):
        super().__init__(problems, device)
        self.seq_len = seq_len
        self.ad_eps = ad_eps

    def __getitem__(self, index: int):
        problem = self.problems[index]

        query_x = torch.randint(0, problem.n, (problem.d,), dtype=torch.float, device=self.device)
        query_y = problem.target(query_x)
        query_state = torch.cat([query_x, query_y.unsqueeze(0)])
        if self.ad_eps:
            # TODO: refine AD+eps logic
            # probes_ratio = 1 - self.ad_max_eps * index / len(self.problems)
            # prob = torch.linspace(0, probes_ratio, self.seq_len)
            # max_probes = torch.multinomial(prob / prob.sum(), num_samples=1).item()
            random_ratio = np.random.uniform(low=self.ad_eps[0], high=self.ad_eps[1], size=(1,))
            num_probes = min(len(problem.x_probes), int(self.seq_len * (1 - random_ratio)))
            probes_idx = torch.randperm(len(problem.x_probes))[:num_probes].tolist()

            x_probes = torch.tensor(problem.x_probes[probes_idx], dtype=torch.float)
            y_probes = torch.tensor(problem.y_probes[probes_idx], dtype=torch.float)
            x_random = torch.randint(0, problem.n, (self.seq_len - num_probes + 1, problem.d), dtype=torch.float, device=self.device)
            y_random = problem.target(x_random)

            states = torch.cat([torch.cat([x_probes, x_random]), torch.cat([y_probes, y_random]).unsqueeze(1)], dim=1)
            states = states[torch.randperm(states.size(0))]
            
            actions = states[1:, :-1]
            next_states = states[1:]
            states = states[:-1]
            target_action = torch.tensor(problem.info["x_min"], dtype=torch.int)
        else:
            x = torch.randint(0, problem.n, (self.seq_len, problem.d), dtype=torch.int, device=self.device)
            actions = torch.randint(0, problem.d + 1, (self.seq_len,), dtype=torch.int, device=self.device)
            
            x_next = x.clone()
            mask = actions < problem.d
            indices = torch.arange(self.seq_len, device=self.device)[mask]
            x_next[indices, actions[mask]] ^= 1

            possible_actions = torch.eye(problem.d + 1, problem.d, dtype=torch.int, device=self.device)
            possible_target_x = possible_actions ^ query_x.long()
            
            y_all = problem.target(torch.cat([x, x_next, possible_target_x]).float()).unsqueeze(1)
            
            states = torch.cat([x, y_all[:x.size(0)]], dim=1)
            next_states = torch.cat([x_next, y_all[x.size(0):x.size(0) + x_next.size(0)]], dim=1)
            target_action = y_all[-possible_target_x.size(0):, 0].argmin()
        
        return {
            "query_state": query_state.float(),
            "states": states.float(),
            "actions": actions.long(),
            "next_states": next_states.float(),
            "target_action": target_action.long(),
        }