import os
from dataclasses import asdict, dataclass
from typing import Optional, Tuple
import random
import numpy as np
import pyrallis
import yaml
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import wandb
from tqdm.autonotebook import tqdm

try:
    from model import DPT_K2D
    from utils.data import MarkovianDataset
    from utils.schedule import cosine_annealing_with_warmup
except ImportError:
    from .model import DPT_K2D
    from .utils.data import MarkovianDataset
    from .utils.schedule import cosine_annealing_with_warmup

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found: {config_path}')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class DPTSolver():
    def __init__(self, config_path):
        self.config = load_config(config_path)['TrainConfig']
        self.model = DPT_K2D(**self.config["model_params"]).to(DEVICE)
        # if needed, test beforehand
        # model = torch.compile(model)
        # print(f"Parameters: {sum(p.numel() for p in self.model.parameters())}")

    def get_batch(self, batch):
        (
            query_states,
            states,
            actions,
            next_states,
            rewards,
            target_actions,
        ) = [b.to(DEVICE) for b in batch]

        query_states = query_states.to(torch.float)
        states = states.to(torch.float)
        actions = actions.to(torch.long)
        next_states = next_states.to(torch.float)
        rewards = rewards.to(torch.float)
    
        target_actions = target_actions.squeeze(-1)
        if self.config["with_prior"]:
            target_actions = (
                F.one_hot(target_actions, num_classes=self.config["model_params"]["num_actions"])
                .unsqueeze(1)
                .repeat(1, self.config["model_params"]["seq_len"] + 1, 1)
                .float()
            )
        else:
            target_actions = (
                F.one_hot(target_actions, num_classes=self.config["model_params"]["num_actions"])
                .unsqueeze(1)
                .repeat(1, self.config["model_params"]["seq_len"], 1)
                .float()
            )
        return (
            query_states,
            states,
            actions,
            next_states,
            rewards,
            target_actions
        )
    
    def get_loss(self, predicted_actions, target_actions):
        loss = F.cross_entropy(
            input=predicted_actions.flatten(0, 1),
            target=target_actions.flatten(0, 1),
            label_smoothing=self.config["label_smoothing"],
        )
        return loss
    
    def get_actions(self, logits):
        temp = 1.0
        temp = 1.0 if temp <= 0 else temp
        probs = torch.nn.functional.softmax(logits / temp, dim=-1)

        do_samples = False
        if do_samples:
            actions = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            actions = torch.argmax(probs, dim=-1)
        return actions

    def get_accuracy(self, predicted_actions, target_actions):
        with torch.no_grad():
            logits = predicted_actions.flatten(0, 1)
            actions = self.get_actions(logits)
            target_actions = torch.argmax(target_actions, dim=-1)
            actions = actions.reshape(target_actions.shape)
            # accuracy = torch.sum(a == t) / np.prod(t.shape)
            accuracy = torch.sum(actions[:, -1] == target_actions[:, -1]) / target_actions.shape[0]
        return accuracy

    def train_step(self, batch):
        with torch.amp.autocast('cuda'):
            (
                query_states,
                states,
                actions,
                next_states,
                rewards,
                target_actions
            ) = self.get_batch(batch)

            predicted_actions = self.model(
                query_states=query_states,
                context_states=states,
                context_actions=actions,
                context_next_states=next_states,
                context_rewards=rewards,
            )

            if not self.config["with_prior"]:
                predicted_actions = predicted_actions[:, 1:, :]

            loss = self.get_loss(predicted_actions, target_actions)

        self.scaler.scale(loss).backward()
        if self.config["clip_grad"] is not None:
            self.scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["clip_grad"])
        self.scaler.step(self.optim)
        self.scaler.update()
        self.optim.zero_grad(set_to_none=True)
        if self.config["with_scheduler"]:
            self.scheduler.step()

        loss = loss.item()
        accuracy = self.get_accuracy(predicted_actions, target_actions)
        return loss, accuracy

    def set_train(self):
        dataset = MarkovianDataset(
            data_path=self.config["learning_histories_path"], 
            seq_len=self.config["model_params"]["seq_len"]
        )
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config["batch_size"],
            pin_memory=True,
            shuffle=False,
            num_workers=self.config["num_workers"],
        )

        self.optim = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
            betas=self.config["betas"],
        )

        current_lr = self.config["learning_rate"]
        if self.config["with_scheduler"]:
            self.scheduler = cosine_annealing_with_warmup(
                optimizer=self.optim,
                warmup_steps=int(self.config["update_steps"] * self.config["warmup_ratio"]),
                total_steps=self.config["update_steps"],
            )
        self.scaler = torch.amp.GradScaler('cuda')

    def train(self, eval_function):
        set_seed(self.config["train_seed"])
        self.set_train()

        wandb.init(
            config=self.config,
            **self.config["wandb_params"]
        )

        for global_step, batch in tqdm(enumerate(self.dataloader), 'Training'):
            if global_step > self.config["update_steps"]:
                break
            loss, accuracy = self.train_step(batch)

            if self.config["with_scheduler"]:
                current_lr = self.scheduler.get_last_lr()[0]

            wandb.log(
                {
                    "loss": loss,
                    "accuracy": accuracy,
                    # "lr": current_lr
                },
                step=global_step,
            )

            if global_step % self.config["eval_every"] == 0:
                eval_info = eval_function(self)
                wandb.log(
                    eval_info,
                    step=global_step,
                )
                # # self.save_model(f"model_{global_step}.pt")
                self.model.train()

        self.save_model(f"model_last.pt")

    def save_model(self, checkpoint_name):
        checkpoint_path = self.config["checkpoints_path"]
        if checkpoint_path is not None:
            torch.save(self.model.state_dict(), os.path.join(checkpoint_path, checkpoint_name))

    def looad_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.model.load_state_dict(checkpoint)
    
    def test(self, query_states, transition_function, n_steps=10):
        """
        transition_function(state, action) = state'
        """
        with torch.no_grad():
            self.model.eval()

            states = torch.Tensor(1, 0)
            next_states = torch.Tensor(1, 0)
            actions = torch.Tensor(1, 0)
            rewards = torch.Tensor(1, 0)

            for _ in range(n_steps):
                predicted_actions = self.model(
                    query_states=query_states.to(dtype=torch.float, device=DEVICE),
                    context_states=states.to(dtype=torch.float, device=DEVICE),
                    context_next_states=next_states.to(dtype=torch.float, device=DEVICE),
                    context_actions=actions.to(dtype=torch.long, device=DEVICE),
                    context_rewards=rewards.to(dtype=torch.float, device=DEVICE),
                )
                predicted_action = self.get_actions(predicted_actions).cpu()[0]
                state = transition_function(query_states, predicted_action)

                states = torch.cat([states, query_states.unsqueeze(0)], dim=1)
                next_states = torch.cat([next_states, state.unsqueeze(0)], dim=1)
                actions = torch.cat([actions, torch.tensor([predicted_action]).unsqueeze(0)], dim=1)
                rewards = torch.cat([rewards, (-1 * (state - query_states)).unsqueeze(0)], dim=1)
                query_states = state
        return (
                query_states,
                states,
                actions,
                next_states,
                rewards
        )