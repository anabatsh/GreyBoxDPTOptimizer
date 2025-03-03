import torch
import torch.nn as nn
from torch.nn import functional as F
import lightning as L

try:
    from heavyball import PaLMForeachSFAdamW
except ImportError:
    PaLMForeachSFAdamW = None

import gc
from tqdm.auto import tqdm

try:
    from model import DPT
    from loss import BCELoss
    from schedule import cosine_annealing_with_warmup
    from reward import Reward
    from metrics import Metrics
except ImportError:
    from .model import DPT
    from .loss import BCELoss
    from .schedule import cosine_annealing_with_warmup
    from .reward import Reward
    from .metrics import Metrics


class DPTSolver(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = BCELoss(config['label_smoothing'])
        self.metrics = Metrics()
        self.reward_model = Reward()
        self.model = DPT(**self.config["model_params"])
        self.save_hyperparameters(ignore=["model, reward_model, loss, metrics"])

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
        batch_size = len(batch['states'])
        outputs = self._offline_step(batch)
        results = self.loss(**outputs) | self.metrics(**outputs)
        for key, val in results.items():
            self.log(f"train_{key}", val, on_step=True, on_epoch=False, batch_size=batch_size)
        return results

    def validation_step(self, batch, batch_idx):
        """
        offline validation step
        """
        batch_size = len(batch['states'])
        outputs = self._offline_step(batch)
        results = self.loss(**outputs) | self.metrics(**outputs)
        for key, val in results.items():
            self.log(f"val_{key}", val, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return results

    def on_test_epoch_start(self):
        self.trajectories = []

    def test_step(self, batch, batch_idx):
        outputs = self._online_step(batch)
        y = outputs["predictions"][..., -1]
        trajectory = torch.cummin(y, -1).values.mean(0)
        if hasattr(self, "trajectories"):
            self.trajectories.append(trajectory)
        return {"y": trajectory[-1]}

    def on_test_epoch_end(self):
        self.trajectory = torch.stack(self.trajectories, dim=0).mean(0)
        return {"y": self.trajectory[-1]}

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

    def _online_step(self, batch): # to fix
        results = self.run(
            query_state=batch["query_state"],
            problems=batch["problem"],
            warmup_steps=self.config["warmup"],
            n_steps=self.config["online_steps"],
            do_sample=self.config["do_sample"],
            temperature_function=lambda x: self.config["temperature"] if self.config["temperature"] is float else self.config["temperature"]  # add support in load_config()
        )
        return {
            "outputs": results["outputs"],
            "targets": batch["target_state"],
            "predictions": results["predictions"],
        }

    def run(self, query_state, problems, warmup_steps=0, n_steps=10, do_sample=False, temperature_function=lambda x: 1.0):
        """
        run an online inference
        """        
        seq_len = self.config["model_params"]["seq_len"]
        state_dim = self.config["model_params"]["state_dim"]
        action_dim = self.config["model_params"]["action_dim"]
        batch_size = len(query_state)

        device = self.device
        total_steps = warmup_steps + n_steps
        states = torch.Tensor(batch_size, total_steps, state_dim).to(dtype=torch.float, device=device)        
        actions = torch.Tensor(batch_size, total_steps, action_dim).to(dtype=torch.long, device=device)
        next_states = torch.Tensor(batch_size, total_steps, state_dim).to(dtype=torch.float, device=device)
        rewards = torch.Tensor(batch_size, total_steps).to(dtype=torch.float, device=device)

        def transition(state, action, problems=problems):
            next_state = state.clone()
            for i in range(batch_size):
                if self.config["parallel"]:
                    next_state[i][:-1] = action[i]
                    next_state[i][-1] = problems[i].target(action[i].float().squeeze().detach())
                else:
                    if action[i] < problems[i].d:
                        next_state[i][action[i]] = torch.abs(1 - next_state[i][action[i]])
                        next_state[i][-1] = problems[i].target(next_state[i][:-1].float().squeeze().detach())
            return next_state

        # preparing warmup context
        for idx in tqdm(range(warmup_steps), total=warmup_steps):
            random_query_state = torch.randint(0, self.config["problem_params"]["n"], query_state.shape, device=device)
            if self.config["parallel"]:
                action = torch.randint(0, self.config["problem_params"]["n"], (batch_size, self.config["problem_params"]["d"]), device=device)
            else:
                action = torch.randint(0, self.config["problem_params"]["n"] + 1, (batch_size, 1), device=device)
                action = torch.eye(self.config["problem_params"]["d"] + 1, self.config["problem_params"]["d"], dtype=torch.int)[actions] 

            next_state = transition(random_query_state, action)

            states[:, idx] = random_query_state
            actions[:, idx] = action
            next_states[:, idx] = next_state
            rewards[:, idx] = self.reward_model.online(
                states=states[:, :idx+1],
                actions=actions[:, :idx+1],
                next_states=next_states[:, :idx+1]
            )

        # optimization loop
        for idx in tqdm(range(n_steps), total=n_steps):
            idx += warmup_steps
            probs = self.model(
                query_state=query_state,
                states=states[:, :idx][:, -seq_len:],
                actions=actions[:, :idx][:, -seq_len:],
                next_states=next_states[:, :idx][:, -seq_len:],
                rewards=rewards[:, :idx][:, -seq_len:]
            )[:, -1, :] 

            prediction = self.get_predictions(
                probs, parallel=self.config["parallel"], 
                do_sample=do_sample, temperature=temperature_function(idx / n_steps)
            )

            # transition
            next_state = transition(query_state, prediction)

            states[:, idx] = query_state
            actions[:, idx] = prediction
            next_states[:, idx] = next_state
            rewards[:, idx] = self.reward_model.online(
                states=states[:, :idx+1],
                actions=actions[:, :idx+1],
                next_states=next_states[:, :idx+1]
            )
            query_state = next_state

        # cleanup
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "outputs": None,
            "predictions": next_states
        }