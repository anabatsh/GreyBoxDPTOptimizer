import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset

root_path = '../'
sys.path.insert(0, root_path)

from scripts.run_solver import load_results


class OnlineDataset(Dataset):
    def __init__(self, problems):
        super().__init__()
        self.problems = problems

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, index: int):
        problem = self.problems[index]

        # query state
        query_x = torch.randint(0, problem.n, (problem.d,), dtype=torch.float)
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

class OfflineDataset(OnlineDataset):
    def __init__(self, problems, seq_len=50, results_dir='', suffix='', ad_ratio=0.0, action='point', target_action='gt'):
        super().__init__(problems)
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

    def get_solver_probes(self, problem, seq_len=10, results_dir='', suffix='test'):
        if seq_len == 0:
            states = torch.tensor([])
        else:
            problem_name = problem.name.split('__seed')[0]
            problem_path = os.path.join(results_dir, problem_name, suffix, problem.name)
            solver_name = np.random.choice(os.listdir(problem_path))
            solver_path = os.path.join(problem_path, solver_name)
            seed_name = np.random.choice(os.listdir(solver_path))
            seed_path = os.path.join(solver_path, seed_name)
            results = load_results(seed_path)
            x = results['x_list']
            y = results['y_list']
            trajectory = torch.cat([x, y.unsqueeze(1)], dim=1)
            n = len(trajectory)
            indexes = np.random.choice(n, min(n, seq_len), replace=False)
            states = trajectory[sorted(indexes)]
        return states

    def get_random_probes(self, problem, seq_len=10):
        if seq_len == 0:
            states = torch.tensor([])
        else:
            x = torch.randint(0, problem.n, (seq_len, problem.d))
            y = problem.target(x)
            states = torch.cat([x, y.unsqueeze(1)], dim=1)
        return states

    def __getitem__(self, index: int):
        # problem = self.problems[index]
        sample = super().__getitem__(index)
        problem = sample["problem"]

        if self.action == 'point':

            if self.target_action == 'max':
                seq_len = self.seq_len + 1
                solver_context = self.get_solver_probes(problem, seq_len, self.results_dir, self.suffix)
                random_context = self.get_random_probes(problem, seq_len - len(context))
                context = torch.cat([solver_context, random_context])[torch.randperm(seq_len)]
                states = context[:-1]
                actions = context[1:, :-1]
                next_states = context[1:]
                target_action = sample["target_state"][:-1]

            # elif self.target_action == 'next':
            #     states = self.get_solver_probes(problem, self.seq_len, self.results_dir, self.suffix)
            #     sample["query_state"] = torch.zeros_like(sample["query_state"])
            #     actions = torch.zeros_like(states[:, :-1])
            #     next_states = torch.zeros_like(states)
            #     target_action = torch.cat([states[1:, :-1], sample["target_state"][:-1].unsqueeze(0)])

            else:
                raise ValueError(f"Invalid target action: {self.target_action} for action {self.action}.")
            
        elif self.action == 'bitflip':
            states = self.get_random_probes(problem, self.seq_len)
            actions = torch.randint(0, problem.d + 1, (self.seq_len,), dtype=torch.int)
            
            x_next = states[:, :-1].long()
            mask = actions < problem.d
            indices = torch.arange(self.seq_len)[mask]
            x_next[indices, actions[mask]] ^= 1
            y_next = problem.target(x_next)
            next_states = torch.cat([x_next, y_next.unsqueeze(1)], dim=1)

            # one-hot encoding
            actions = torch.eye(problem.d + 1, problem.d + 1, dtype=torch.int)[actions]

            if self.target_action == 'greedy':
                query_x = sample["query_state"][:-1].long()                
                possible_actions = torch.eye(problem.d + 1, problem.d, dtype=torch.int)
                possible_target_x = possible_actions ^ query_x
                possible_target_y = problem.target(possible_target_x)
                target_action = possible_target_y.argmin()
                target_action = torch.eye(problem.d + 1, problem.d + 1, dtype=torch.int)[target_action]
            
            elif self.target_action == 'gt':
                target_action = torch.randint(0, problem.d + 1, (1,), dtype=torch.int)[0]

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
                target_action = torch.eye(problem.d + 1, problem.d + 1, dtype=torch.int)[target_action]

            elif self.target_action == 'gt_greedy':
                target_x = sample["target_state"][:-1].long()
                query_x = sample["query_state"][:-1].long()
                target_actions = target_x ^ query_x
                possible_actions = torch.eye(problem.d).long()[target_actions.to(torch.bool)]
                possible_actions = torch.cat([possible_actions, torch.zeros(1, possible_actions.shape[-1], dtype=torch.int)], dim=0)
                possible_target_x = possible_actions ^ query_x
                possible_target_y = problem.target(possible_target_x)
                target_action = possible_target_y.argmin()
                target_action = possible_actions[target_action]
                last_bit = int(target_action.sum() == 0)
                target_action = torch.cat([target_action, torch.tensor([last_bit])], dim=0)

            elif self.target_action == 'gt_multi':
                target_x = sample["target_state"][:-1].long()
                query_x = sample["query_state"][:-1].long()
                target_action = target_x ^ query_x
                last_bit = int(target_action.sum() == 0)
                target_action = torch.cat([target_action, torch.tensor([last_bit])], dim=0)

            elif self.target_action == 'gt_ad':
                context = self.get_solver_probes(problem, self.seq_len+1, self.results_dir, self.suffix)
                states = context[:-1]
                next_states = context[1:]
                actions = context[1:, :-1].long() ^ context[:-1, :-1].long()
                last_bit = (actions.sum(-1) == 0).long()
                actions = torch.cat([actions, last_bit.unsqueeze(-1)], dim=-1)

                target_x = sample["target_state"][:-1].long()
                query_x = sample["query_state"][:-1].long()
                target_action = target_x ^ query_x
                last_bit = int(target_action.sum() == 0)
                target_action = torch.cat([target_action, torch.tensor([last_bit])], dim=0)

            elif self.target_action == 'gt_ad_greedy':
                context = self.get_solver_probes(problem, self.seq_len+1, self.results_dir, self.suffix)
                states = context[:-1]
                next_states = context[1:]
                actions = context[1:, :-1].long() ^ context[:-1, :-1].long()
                last_bit = (actions.sum(-1) == 0).long()
                actions = torch.cat([actions, last_bit.unsqueeze(-1)], dim=-1)

                target_x = sample["target_state"][:-1].long()
                query_x = sample["query_state"][:-1].long()
                target_actions = target_x ^ query_x
                possible_actions = torch.eye(problem.d).long()[target_actions.to(torch.bool)]
                possible_actions = torch.cat([possible_actions, torch.zeros(1, possible_actions.shape[-1], dtype=torch.int)], dim=0)
                possible_target_x = possible_actions ^ query_x
                possible_target_y = problem.target(possible_target_x)
                target_action = possible_target_y.argmin()
                target_action = possible_actions[target_action]
                last_bit = int(target_action.sum() == 0)
                target_action = torch.cat([target_action, torch.tensor([last_bit])], dim=0)

            else:
                raise ValueError(f"Invalid target action: {self.target_action}")
            
        else:
            raise ValueError(f"Invalid action: {self.action}")

        if target_action.ndim == 1:
            target_action = target_action.repeat(self.seq_len + 1, 1)

        sample |= {
            "states": states.float(),
            "actions": actions.long(),
            "next_states": next_states.float(),
            "target_action": target_action.long()
        }
        return sample