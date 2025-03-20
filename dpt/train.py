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
    from loss import BCELoss, CELoss, RKLLoss
    from schedule import cosine_annealing_with_warmup
    from reward import Reward, ZeroReward
    from metrics import PointMetrics, BitflipMetrics
except ImportError:
    from .model import DPT
    from .loss import BCELoss, CELoss, RKLLoss
    from .schedule import cosine_annealing_with_warmup
    from .reward import Reward, ZeroReward
    from .metrics import PointMetrics, BitflipMetrics


class DPTSolver(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        action = config['action']
        if action == 'point':
            self.loss = CELoss(config['label_smoothing']) #BCELoss(config['label_smoothing'])
            self.metrics = PointMetrics()
        elif action == 'bitflip':
            self.loss = RKLLoss(config['label_smoothing']) #CELoss(config['label_smoothing'])
            self.metrics = BitflipMetrics()
        else:
            raise ValueError(f"Unknown action type: {action}")
        
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

    def get_predictions(self, outputs, do_sample=False, temperature=1.0):
        """
        outputs - [batch_size, seq_len + 1, action_dim]
        """
        action = self.config["action"]
        if action == "point":
            if do_sample and temperature > 0:
                probs = torch.sigmoid(outputs / temperature)
                predictions = torch.bernoulli(probs)
            else:
                predictions = (torch.sigmoid(outputs) > 0.5).long()
        elif action == "bitflip":
            if do_sample and temperature > 0:
                probs = F.softmax(outputs / temperature, dim=-1)
                predictions = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                predictions = torch.argmax(outputs, dim=-1)
            d = outputs.shape[-1]
            predictions = torch.eye(d, d, dtype=torch.int, device=predictions.device)[predictions]
        else:
            raise ValueError(f"Unknown action type: {action}")
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
            "predictions": self.get_predictions(outputs),
        }

    def _online_step(self, batch):
        results = self.run(
            query_state=batch["query_state"],
            problems=batch["problem"]
        )
        return {
            "outputs": results["outputs"],
            "targets": batch["target_state"],
            "predictions": results["predictions"],
        }

    def run(self, query_state, problems):
        """
        run an online inference
        """
        seq_len = self.config["model_params"]["seq_len"]
        state_dim = self.config["model_params"]["state_dim"]
        action_dim = self.config["model_params"]["action_dim"]
        warmup_steps = self.config["warmup_steps"]
        n_steps = self.config["online_steps"]
        do_sample = self.config["do_sample"]
        temperature_function = lambda x: self.config["temperature"] if self.config["temperature"] is float else self.config["temperature"]  # add support in load_config()
        action_mode = self.config["action"]
        # assert action_mode in ["point", "bitflip"], f"Unknown action mode: {action_mode}"

        device = self.device
        batch_size = len(query_state)
        total_steps = warmup_steps + n_steps

        states = torch.Tensor(batch_size, total_steps, state_dim).to(dtype=torch.float, device=device)        
        actions = torch.Tensor(batch_size, total_steps, action_dim).to(dtype=torch.long, device=device)
        next_states = torch.Tensor(batch_size, total_steps, state_dim).to(dtype=torch.float, device=device)
        rewards = torch.Tensor(batch_size, total_steps).to(dtype=torch.float, device=device)

        def transition(state, action, problems=problems):
            next_state = state.clone()
            for i in range(batch_size):
                if action_mode == "point":
                    next_state[i][..., :-1] = action[i]
                elif action_mode == "bitflip":
                    next_state[i][..., :-1] = state[i][..., :-1].long() ^ action[i][..., :-1]
                else:
                    raise ValueError(f"Unknown action mode: {action_mode}")
                next_state[i][..., -1] = problems[i].target(next_state[i][..., :-1].float().squeeze().detach())
            return next_state

        # preparing warmup context        
        if warmup_steps > 0:
            warmup_x = torch.randint(0, self.config["problem_params"]["n"], (batch_size, warmup_steps, self.config["problem_params"]["d"]), device=device)
            warmup_y = torch.stack([problem.target(x) for x, problem in zip(warmup_x, problems)], dim=0)
            warmup_states = torch.cat([warmup_x, warmup_y.unsqueeze(-1)], dim=-1)

            if action_mode == "point":
                warmup_actions = torch.randint(0, self.config["problem_params"]["n"], (batch_size, warmup_steps, self.config["problem_params"]["d"]), device=device)
            elif action_mode == "bitflip":
                warmup_actions = torch.randint(0, self.config["problem_params"]["n"] + 1, (batch_size, warmup_steps), device=device)
                warmup_actions = F.one_hot(warmup_actions, self.config["problem_params"]["d"] + 1)
            else:
                raise ValueError(f"Unknown action mode: {action_mode}")

            warmup_next_states = transition(warmup_states, warmup_actions)
            warmup_rewards = self.reward_model.offline(
                states=warmup_states,
                actions=warmup_actions,
                next_states=warmup_next_states
            )
            states[:, :warmup_steps] = warmup_next_states
            actions[:, :warmup_steps] = warmup_actions
            next_states[:, :warmup_steps] = warmup_next_states
            rewards[:, :warmup_steps] = warmup_rewards

        # optimization loop
        for idx in tqdm(range(n_steps), total=n_steps, desc='Online', leave=False):
            idx += warmup_steps
            probs = self.model(
                query_state=query_state,
                states=states[:, :idx][:, -seq_len:],
                actions=actions[:, :idx][:, -seq_len:],
                next_states=next_states[:, :idx][:, -seq_len:],
                rewards=rewards[:, :idx][:, -seq_len:]
            )[:, -1, :] 

            temperature = temperature_function(idx / n_steps)
            prediction = self.get_predictions(probs, do_sample, temperature)

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