import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import lightning as L
from collections import defaultdict
try:
    from model import DPT
    from schedule import cosine_annealing_with_warmup
except ImportError:
    from .model import DPT
    from .schedule import cosine_annealing_with_warmup


def relative_improvement(x, y):
    return np.abs(x - y) / (np.finfo(x.dtype).eps + np.abs(x))


class DPTSolver(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = DPT(**config["model_params"])
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            **self.config["optimizer_params"]
        )
        if self.config["with_scheduler"]:
            scheduler = cosine_annealing_with_warmup(
                optimizer=optimizer,
                total_epochs=self.config["max_epochs"],
                **self.config["scheduler_params"]
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer
    
    def training_step(self, batch, batch_idx):
        """
        offline training step
        """
        outputs = self._offline_step(batch)
        results = self.get_loss(outputs) | self.get_metrics(outputs)
        for key, val in results.items():
            self.log(f"train {key}", val, on_step=True, on_epoch=False)
        return results

    def validation_step(self, batch, batch_idx):
        """
        offline validation step
        """
        outputs = self._offline_step(batch)
        results = self.get_loss(outputs) | self.get_metrics(outputs)
        for key, val in results.items():
            self.log(f"val {key}", val, on_step=False, on_epoch=True)
        return results

    def on_test_epoch_start(self):
        self.trajectories = []

    def test_step(self, batch, batch_idx):
        """
        online test step
        """
        outputs = self._online_step(batch)
        trajectory = outputs["trajectory"]
        trajectory = torch.cummin(trajectory[..., -1], dim=1).values
        if hasattr(self, "trajectories"):
            self.trajectories.append(trajectory.mean(0))
        return {"problems": batch["problem"], "y": trajectory}

    def on_test_epoch_end(self):
        self.trajectory = torch.vstack(self.trajectories).mean(0)
        self.trajectories.clear()
        return {"y": self.trajectory[-1]}

    def get_loss(self, outputs):
        """
        outputs - [batch_size, seq_len + 1, action_dim]
        targets - [batch_size]
        """
        log_outputs = F.log_softmax(outputs["outputs"], -1).permute(0, 2, 1)
        targets = outputs["targets"][:, None].repeat(1, log_outputs.shape[-1])
        loss = F.nll_loss(log_outputs[..., 1:], targets[..., 1:])
        return {"loss": loss}

    def get_metrics(self, outputs):
        """
        predictions - [batch_size, seq_len + 1]
        targets     - [batch_size]
        """
        predictions = outputs["predictions"]
        targets = outputs["targets"]
        accuracy = (predictions == targets[:, None]).to(torch.float).mean()
        mae = torch.abs(predictions - targets[:, None]).to(torch.float).mean()
        return {"accuracy": accuracy, "mae": mae}

    def get_predictions(self, outputs, do_sample=False, temperature=1.0):
        """
        outputs - [batch_size, seq_len + 1, action_dim + 2]
        """
        if do_sample and temperature > 0:
            probs = F.softmax(outputs / temperature, dim=-1)
            predictions = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            predictions = torch.argmax(outputs, dim=-1)
        return predictions
    
    def _offline_step(self, batch):
        outputs = self.model(
            query_state=batch["query_state"],
            states=batch["states"],
            actions=batch["actions"],
            next_states=batch["next_states"],
            rewards=batch["rewards"]
        )
        predictions = self.get_predictions(outputs)
        return {
            "outputs": outputs,
            "predictions": predictions,
            "targets": batch["target_action"]
        }

    def _online_step(self, batch):
        trajectory = []
        try:
            for query_state, context, problem in zip(
                batch["query_state"], 
                zip(batch["states"], batch["actions"], batch["next_states"], batch["rewards"]),
                batch["problem"]
            ):
                results = self.run(
                    query_state=query_state,
                    context=context,
                    problem=problem,
                    n_steps=self.config["online_steps"],
                    do_sample=self.config["do_sample"],
                    temperature_function=lambda x: self.config["temperature"]
                )
                trajectory.append(results["next_states"])
        except:
            for query_state, problem in zip(
                batch["query_state"], 
                batch["problem"]
            ):
                results = self.run(
                    query_state=query_state,
                    context=[],
                    problem=problem,
                    n_steps=self.config["online_steps"],
                    do_sample=self.config["do_sample"],
                    temperature_function=lambda x: self.config["temperature"]
                )
                trajectory.append(results["next_states"])

        return {"trajectory": torch.stack(trajectory)}

    def run(self, query_state, context, problem, n_steps=10, do_sample=False, temperature_function=lambda x: 1.0):
        """
        run an online inference
        """
        device = query_state.device
        seq_len = self.config["model_params"]["seq_len"]
        state_dim = self.config["model_params"]["state_dim"]
        action_dim = self.config["model_params"]["action_dim"]

        if len(context):
            # [1, state_dim]
            query_state = query_state.unsqueeze(0)
            # [1, 0, state_dim]
            states = context[0].unsqueeze(0)
            # [1, 0]
            actions = context[1].unsqueeze(0)
            # [1, 0, state_dim]
            next_states = context[2].unsqueeze(0)
            # [1, 0]
            rewards = context[3].unsqueeze(0)
        else:
            # [1, state_dim]
            query_state = query_state.unsqueeze(0)
            # [1, 0, state_dim]
            states = torch.Tensor(1, 0, state_dim).to(dtype=torch.float, device=device)
            # [1, 0]
            actions = torch.Tensor(1, 0).to(dtype=torch.long, device=device)
            # [1, 0, state_dim]
            next_states = torch.Tensor(1, 0, state_dim).to(dtype=torch.float, device=device)
            # [1, 0]
            rewards = torch.Tensor(1, 0).to(dtype=torch.float, device=device)

        for n_step in range(n_steps):
            # [1, action_dim]
            output = self.model(
                query_state=query_state,
                states=states[:, -seq_len:],
                actions=actions[:, -seq_len:],
                next_states=next_states[:, -seq_len:],
                rewards=rewards[:, -seq_len:]
            )[:, -1, :]
            # [1]
            action = self.get_predictions(output, do_sample=do_sample, temperature=temperature_function(n_step))
            # [1, state_dim]
            next_state = query_state.clone()
            if action[0] < problem.d:
                next_state[0][action[0]] = torch.abs(1 - next_state[0][action[0]])
                next_state[0][-1] = problem.target(next_state[0][:-1].cpu().detach().numpy())
            # [1, n_step, state_dim]
            states = torch.cat([states, query_state.unsqueeze(1)], dim=1)
            # [1, n_step]
            actions = torch.cat([actions, action.unsqueeze(1)], dim=1)
            # [1, n_step, state_dim]
            next_states = torch.cat([next_states, next_state.unsqueeze(1)], dim=1)
            # [1]
            # reward = torch.tensor([0.0], device=device)
            y = next_states[..., -1].cpu().detach().numpy()
            y = np.hstack((y[:, [0]], y))

            alpha = 0.5
            reward_exploit = relative_improvement(y[:, :-1].min(1), y[:, -1])
            exploration = np.abs(y[:, :-1] - y[:, -1:])
            reward_explore = 1 / (1 + exploration.min(1))
            reward = torch.tensor(alpha * reward_exploit + (1 - alpha) * reward_explore, device=device)

            # [1, n_step]
            rewards = torch.cat([rewards, reward.unsqueeze(1)], dim=1)
            # [1, n_step, action_dim + 2]
            # outputs = torch.cat([outputs, output.unsqueeze(1)], dim=1)

            query_state = next_state

        return {
            "query_state": query_state[0],
            "states": states[0],
            "actions": actions[0],
            "next_states": next_states[0],
            "rewards": rewards[0],
            # "outputs": outputs[0],
        }