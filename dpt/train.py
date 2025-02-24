import torch
import torch.nn as nn
from torch.nn import functional as F
import lightning as L
from collections import defaultdict
try:
    from model import DPT
    from schedule import cosine_annealing_with_warmup
    from reward import Reward
except ImportError:
    from .model import DPT
    from .schedule import cosine_annealing_with_warmup
    from .reward import Reward
try:
    from heavyball import PaLMForeachSFAdamW
except ImportError:
    PaLMForeachSFAdamW = None

from tqdm.auto import tqdm
import gc

class DPTSolver(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.reward_model = Reward()
        self.save_hyperparameters()
        self.model = None

    def configure_model(self):
        if self.model is not None:
            return
        self.model = DPT(**self.config["model_params"])

    def configure_optimizers(self):
        opt_cls = PaLMForeachSFAdamW if PaLMForeachSFAdamW is not None else torch.optim.AdamW
        optimizer = opt_cls(
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
        # batch["rewards"] = self.reward_model()
        outputs = self._offline_step(batch)
        results = self.get_loss(**outputs) | self.get_metrics(**outputs)
        for key, val in results.items():
            self.log(f"train_{key}", val, on_step=True, on_epoch=False)
        return results

    def validation_step(self, batch, batch_idx):
        """
        offline validation step
        """
        outputs = self._offline_step(batch)
        results = self.get_loss(**outputs) | self.get_metrics(**outputs)
        for key, val in results.items():
            self.log(f"val_{key}", val, on_step=False, on_epoch=True, sync_dist=True)
        return results

    def on_test_epoch_start(self):
        self.trajectories = []

    def test_step(self, batch, batch_idx):
        outputs = self._online_step(batch)
        if hasattr(self, "trajectories"):
            self.trajectories.append(outputs["best_predictions"][..., -1])
        return {"problems": batch["problem"], "y": outputs["best_predictions"][:, -1, -1]}

    def on_test_epoch_end(self):
        self.trajectory = torch.cat(self.trajectories).mean(0)
        # self.trajectories.clear()
        return {"y": self.trajectory[-1]}

    def get_loss(self, outputs, targets, predictions):
        """
        outputs - [batch_size, seq_len + 1, action_dim]
        targets - [batch_size] or [batch_size, action_dim]
        """
        if targets.dim() == 2 and targets.size(1) == outputs.size(-1):
            # Binary Cross-Entropy for parallel tasks
            loss_fct = nn.BCEWithLogitsLoss()

            # Reshape outputs to [batch_size * (seq_len + 1), action_dim]
            logits = outputs.view(-1, outputs.size(-1)).contiguous()

            # Repeat targets to match the sequence length dimension
            # targets will be [batch_size, seq_len + 1, action_dim] after repeat
            targets = targets.float().unsqueeze(1).repeat(1, outputs.size(1), 1)

            # Reshape targets to [batch_size * (seq_len + 1), action_dim]
            labels = targets.view(-1, targets.size(-1)).contiguous()

            # Apply label smoothing
            if self.config["label_smoothing"] > 0:
                smoothing = self.config["label_smoothing"]
                labels = labels * (1 - smoothing) + 0.5 * smoothing  # Smooth labels towards 0.5
        else:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=self.config["label_smoothing"])

            # Reshape outputs to [batch_size * (seq_len + 1), action_dim]
            logits = outputs.view(-1, outputs.size(-1)).contiguous()

            # Repeat targets to match the sequence length dimension
            # targets will be [batch_size, seq_len + 1] after repeat
            labels = targets.unsqueeze(1).repeat(1, outputs.size(1)).view(-1).long().contiguous()

        loss = loss_fct(logits, labels)
        return {"loss": loss}

    def get_metrics(self, outputs, targets, predictions, problem=None):
        """
        offline mode:
            predictions - [batch_size, seq_len + 1]
            targets     - [batch_size]
        online mode:
            predictions - [batch_size, state_dim] or [batch_size, seq_len + 1, state_dim]
            targets     - [batch_size, state_dim]
        """
        if targets.ndim == 2 and targets.size(1) == self.config["model_params"]["state_dim"]:
            targets = targets.float()
            if predictions.ndim == 2:
                x_mae = torch.abs(predictions[:, :-1] - targets[:, :-1]).sum(-1).mean()
                y_mae = torch.abs(predictions[:, -1] - targets[:, -1]).mean()
                return {"x_mae": x_mae, "y_mae": y_mae}
            else:
                x_mae = torch.abs(predictions[:, :, :-1] - targets[:, None, :-1]).sum(-1).mean(0)
                y_mae = torch.abs(predictions[:, :, -1] - targets[:, None, -1]).mean(0)
                return {"x_mae": x_mae, "y_mae": y_mae}

        targets = targets.long()
        if targets.ndim == 1:
            accuracy = (predictions == targets[:, None, None]).float()
            mae = torch.abs(predictions - targets[:, None, None]).float()
        else:
            accuracy = (predictions == targets[:, None]).float()
            mae = torch.abs(predictions - targets[:, None]).float()
        return {
            "accuracy": accuracy.mean(), 
            "accuracy_last": accuracy[:, -1].mean(),
            "x_mae": mae.mean(),
            "x_mae_last": mae[:, -1].mean(),
        }

    def get_predictions(self, outputs, do_sample=False, temperature=1.0, parallel=False):
        """
        outputs - [batch_size, seq_len + 1, action_dim]
        parallel: If True, treat as a parallel (multi-label) task.
        """
        if parallel:
            # Single-label binary task
            if do_sample and temperature > 0:
                probs = torch.sigmoid(outputs / temperature)
                predictions = torch.bernoulli(probs)
            else:
                predictions = (torch.sigmoid(outputs) > 0.5).long()
        else:
            if do_sample and temperature > 0:
                probs = F.softmax(outputs / temperature, dim=-1)
                predictions = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                predictions = torch.argmax(outputs, dim=-1)
        return predictions

    def _offline_step(self, batch):
        rewards = self.reward_model.offline(
            states=batch["states"],
            actions=batch["actions"],
            next_states=batch["next_states"]
        )
        outputs = self.model(
            query_state=batch["query_state"],
            states=batch["states"],
            actions=batch["actions"],
            next_states=batch["next_states"],
            rewards=rewards,
        )
        return {
            "outputs": outputs,
            "targets": batch["target_action"],
            "predictions": self.get_predictions(outputs, parallel=self.config["parallel"]),
        }

    def _online_step(self, batch):
        results = self.run(
            query_state=batch["query_state"],
            problems=batch["problem"],
            n_steps=self.config["online_steps"],
            do_sample=self.config["do_sample"],
            temperature_function=lambda x: self.config["temperature"]
        )
        return {
            "outputs": results["outputs"],
            "all_predictions": results["next_states"],
            "best_predictions": results["best_states"],
        }

    def run(self, query_state, problems, warmup=50, n_steps=10, do_sample=False, temperature_function=lambda x: 1.0):
        """
        run an online inference
        """
        device = query_state.device
        if self.config["model_params"]["warmup"] is not None:
            warmup = self.config["model_params"]["warmup"]
        if self.config["temperature"] is not None:
            if callable(self.config["temperature"]) and self.config["temperature"].__name__ == "<lambda>":
                temperature_function = self.config["temperature"]
            else:
                temperature_function = lambda x: self.config["temperature"]
        seq_len = self.config["model_params"]["seq_len"]
        state_dim = self.config["model_params"]["state_dim"]
        action_dim = self.config["model_params"]["action_dim"]
        if query_state.ndim == 1:
            # [1, state_dim]
            query_state = query_state.unsqueeze(0)
            problems = [problems]
        batch_size = query_state.size(0)
        # [batch_size, 0, state_dim]
        states = torch.Tensor(batch_size, 0, state_dim).to(dtype=torch.float, device=device)
        # [batch_size, 0]
        if self.config["parallel"]:
            actions = torch.Tensor(batch_size, 0, action_dim).to(dtype=torch.long, device=device)
        else:
            actions = torch.Tensor(batch_size, 0).to(dtype=torch.long, device=device)
        # [batch_size, 0, state_dim]
        next_states = torch.Tensor(batch_size, 0, state_dim).to(dtype=torch.float, device=device)
        # [batch_size, 0, 1]
        rewards = torch.Tensor(batch_size, 0).to(dtype=torch.float, device=device)
        # [batch_size, 0, action_dim]
        outputs = torch.Tensor(batch_size, 0, action_dim).to(dtype=torch.float, device=device)
        if warmup:
            # print(f"preparing warmup context of {warmup} states")
            warmup_x, warmup_y = [], []
            for i in range(batch_size):
                x = torch.randint(0, problems[i].n, dtype=torch.float, device=states.device, size=(warmup, problems[i].d))
                y = problems[i].target(x)
                warmup_x.append(x)
                warmup_y.append(y)
            warmup_states = torch.cat([torch.stack(warmup_x), torch.stack(warmup_y)[:, :, None]], dim=-1)
        for n_step in tqdm(range(n_steps + warmup), position=0):
            # [batch_size, state_dim]
            if n_step < warmup:
                query_state = warmup_states[:, n_step]
                if self.config["parallel"]:
                    output = torch.randint(0, self.config["problem_params"]["n"], (batch_size, self.config["problem_params"]["d"]))
                else:
                    output = torch.randint(0, self.config["problem_params"]["d"] + 1, (batch_size,))
                output = output.to(device)
                predicted_action = output.clone()
                if not self.config["parallel"]:
                    output = F.one_hot(output, num_classes=action_dim)
            else:
                output = self.model(
                    query_state=query_state,
                    states=states[:, -seq_len:],
                    actions=actions[:, -seq_len:],
                    next_states=next_states[:, -seq_len:],
                    rewards=rewards[:, -seq_len:]
                )[:, -1, :] # type: ignore

                predicted_action = self.get_predictions(output, parallel=self.config["parallel"], do_sample=do_sample, 
                                                        temperature=temperature_function(n_step / (n_steps + warmup)))

            # [batch_size, state_dim]
            predicted_state = query_state.clone()
            for i in range(batch_size):
                if self.config["parallel"]:
                    predicted_state[i][:-1] = predicted_action[i]
                    predicted_state[i][-1] = problems[i].target(predicted_action[i].float().squeeze().detach())
                else:
                    if predicted_action[i] < problems[i].d:
                        predicted_state[i][predicted_action[i]] = torch.abs(1 - predicted_state[i][predicted_action[i]])
                        predicted_state[i][-1] = problems[i].target(predicted_state[i][:-1].float().squeeze().detach())
            # [batch_size, n_step, state_dim]
            states = torch.cat([states, query_state.unsqueeze(1)], dim=1)
            # [batch_size, n_step]
            actions = torch.cat([actions, predicted_action.unsqueeze(1)], dim=1)
            # [batch_size, n_step, state_dim]
            next_states = torch.cat([next_states, predicted_state.unsqueeze(1)], dim=1)
            # [batch_size, n_step, state_dim]
            outputs = torch.cat([outputs, output.unsqueeze(1)], dim=1)  if outputs.size(1) else output.unsqueeze(1)
            reward = self.reward_model.online(
                states=states,
                actions=actions,
                next_states=next_states
            )
            if rewards.size(1):
                reward += rewards[:, -1] # Reward-To-Go
            # [batch_size, n_step]
            rewards = torch.cat([rewards, reward.unsqueeze(1)], dim=1)
            query_state = predicted_state

        # [batch_size, n_step, state_dim]
        y = next_states[..., -1].squeeze(-1)
        best_ys = y.cummin(1)[1]
        best_states = torch.gather(next_states, dim=1, index=best_ys.unsqueeze(-1).expand(-1, -1, next_states.size(-1)))
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "query_state": query_state,
            "states": states,
            "actions": actions,
            "next_states": next_states,
            "rewards": rewards,
            "outputs": outputs,
            "best_states": best_states,
        }