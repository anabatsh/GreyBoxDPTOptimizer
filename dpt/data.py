import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset

root_path = '../'
sys.path.insert(0, root_path)

from scripts.run_solver import load_results


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
        query_x = torch.randint(0, problem.n, (problem.d,), dtype=torch.float, device=self.device)
        query_y = problem.target(query_x)
        query_state = torch.cat([query_x, query_y.unsqueeze(0)])

        # target_state
        if problem.info is None:
            target_state = query_state
        else:
            target_x = problem.info["x_best"]
            target_y = problem.info["y_best"]
            target_state = torch.cat([target_x, target_y.unsqueeze(0)])

        return {
            "query_state": query_state.float(),
            "target_state": target_state.float(),
            "problem": problem,
        }

def get_solver_probes(problem, seq_len=10, results_dir='', suffix='test'):
    if seq_len == 0:
        x = torch.tensor([])
        y = torch.tensor([])
    else:
        problem_class = problem.name.split('__seed')[0]
        problem_path = os.path.join(results_dir, problem_class, suffix)
        solvers = [solver[:-5] for solver in os.listdir(problem_path)]
        solver = np.random.choice(solvers)
        solver_path = os.path.join(problem_path, solver)
        results = load_results(solver_path)[problem.name]
        seed = np.random.choice(len(results['y_best']))
        indexes = np.random.choice(len(results['m_list'][seed]), seq_len, replace=False)
        x = torch.tensor([results['x_list'][seed][i] for i in indexes]).float()
        y = torch.tensor([results['y_list'][seed][i] for i in indexes]).float()
    return x, y

def get_random_probes(problem, seq_len=10):
    if seq_len == 0:
        x = torch.tensor([])
        y = torch.tensor([])
    else:
        x = torch.randint(0, problem.n, (seq_len, problem.d))
        y = problem.target(x)
    return x, y

class OfflineDataset(OnlineDataset):
    def __init__(self, problems, seq_len=50, results_dir='', suffix='', ad_ratio=0.0, action='point', target_action='gt', device='cpu'):
        super().__init__(problems, device)
        self.seq_len = seq_len
        self.results_dir = results_dir
        self.suffix = suffix
        self.ad_ratio = ad_ratio
        self.action = action
        self.target_action = target_action

        if action == 'bitflip' and ad_ratio > 0:
            raise ValueError("Bitflip action is not supported with AD yet")
        
        if action == 'point' and target_action == 'greedy':
            raise ValueError("Greed search is not supported with point action")

    def __getitem__(self, index: int):
        # problem = self.problems[index]
        sample = super().__getitem__(index)
        problem = sample["problem"]

        if self.action == 'point':
            seq_len = self.seq_len + 1
            n_solver = int(self.ad_ratio * seq_len)
            n_random = seq_len - n_solver
            x_solver, y_solver = get_solver_probes(problem, n_solver, self.results_dir, self.suffix)
            x_random, y_random = get_random_probes(problem, n_random)
            x = torch.cat([x_solver, x_random])
            y = torch.cat([y_solver, y_random])
            context = torch.cat([x, y.unsqueeze(1)], dim=1)
            context = context[torch.randperm(context.size(0))]

            states = context[:-1]
            actions = context[1:, :-1]
            next_states = context[1:]
            target_action = sample["target_state"][:-1]

        elif self.action == 'bitflip':
            x, y = get_random_probes(problem, self.seq_len)
            states = torch.cat([x, y.unsqueeze(1)], dim=1)

            actions = torch.randint(0, problem.d + 1, (self.seq_len,), dtype=torch.int, device=self.device)
            
            x_next = x.clone()
            mask = actions < problem.d
            indices = torch.arange(self.seq_len, device=self.device)[mask]
            x_next[indices, actions[mask]] ^= 1
            y_next = problem.target(x_next)
            next_states = torch.cat([x_next, y_next.unsqueeze(1)], dim=1)

            if self.target_action == 'gt':
                target_action = torch.randint(0, problem.d + 1, (1,), dtype=torch.int, device=self.device)[0]

                target_x = sample["target_state"][:-1].long()
                query_x = sample["query_state"][:-1].long()
                if target_action < problem.d:
                    query_x[:target_action] = target_x[:target_action]
                    query_x[target_action] = target_x[target_action] ^ 1
                else:
                    query_x = target_x
                query_y = problem.target(query_x)
                query_state = torch.cat([query_x, query_y.unsqueeze(0)])
                sample["query_state"] = query_state.float()

            elif self.target_action == 'greedy':
                query_x = sample["query_state"][:-1].long()                
                possible_actions = torch.eye(problem.d + 1, problem.d, dtype=torch.int, device=self.device)
                possible_target_x = possible_actions ^ query_x
                possible_target_y = problem.target(possible_target_x)
                target_action = possible_target_y.argmin()
            else:
                raise ValueError(f"Invalid target action: {self.target_action}")
        else:
            raise ValueError(f"Invalid action: {self.action}")
        
        sample |= {
            "states": states.float(),
            "actions": actions.long(),
            "next_states": next_states.float(),
            "target_action": target_action.long()
        }
        return sample

        #     # TODO: refine AD+eps logic
        #     # probes_ratio = 1 - self.ad_max_eps * index / len(self.problems)
        #     # prob = torch.linspace(0, probes_ratio, self.seq_len)
        #     # max_probes = torch.multinomial(prob / prob.sum(), num_samples=1).item()
        #     random_ratio = np.random.uniform(low=self.ad_eps[0], high=self.ad_eps[1], size=(1,))
        #     num_probes = min(len(problem.x_probes), int(self.seq_len * (1 - random_ratio)))
        #     probes_idx = torch.randperm(len(problem.x_probes))[:num_probes].tolist()
