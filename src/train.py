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
        results = self.get_loss(**outputs) | self.get_metrics(**outputs)
        for key, val in results.items():
            self.log(f"train {key}", val, on_step=True, on_epoch=False)
        return results

    def validation_step(self, batch, batch_idx):
        """
        offline validation step
        """
        outputs = self._offline_step(batch)
        results = self.get_loss(**outputs) | self.get_metrics(**outputs)
        for key, val in results.items():
            self.log(f"val {key}", val, on_step=False, on_epoch=True)
        return results

    def on_test_epoch_start(self):
        self.results = defaultdict(list)

    def test_step(self, batch, batch_idx):
        """
        online test step
        """
        outputs = self._online_step(batch)
        results = self.get_metrics(
            outputs=outputs["outputs"], 
            targets=outputs["targets"], 
            predictions=outputs["best_prediction"]
        )
        for key, val in results.items():
            self.log(f"test {key}", val, on_step=False, on_epoch=True)

        if hasattr(self, "results"):
            all_predictions_results = self.get_metrics(
                outputs=outputs["outputs"], 
                targets=outputs["targets"], 
                predictions=outputs["all_predictions"]
            )
            for key, val in all_predictions_results.items():
                self.results[key].append(val)
        return results
    
    def on_test_epoch_end(self):
        results = {
            key: torch.vstack(val).mean(0) 
            for key, val in self.results.items()
        }
        self.save_results = results
        # self.results.clear()
        return results

    def get_loss(self, outputs, targets, predictions):
        """
        outputs - [batch_size, seq_len + 1, action_dim]
        targets - [batch_size]
        """
        log_outputs = F.log_softmax(outputs, -1).permute(0, 2, 1)
        targets = targets[:, None].repeat(1, outputs.shape[1])
        loss = F.nll_loss(log_outputs[..., 1:], targets[..., 1:])
        return {"loss": loss}
    
    def get_metrics(self, outputs, targets, predictions):
        """
        offline mode:
            predictions - [batch_size, seq_len + 1]
            targets     - [batch_size]
        online mode:
            predictions - [batch_size, state_dim] or [batch_size, seq_len + 1, state_dim]
            targets     - [batch_size, state_dim]
        """
        if targets.ndim == 2:
            if predictions.ndim == 2:
                x_mae = torch.abs(predictions[:, :-1] - targets[:, :-1]).sum(-1).mean()
                y_mae = torch.abs(predictions[:, -1] - targets[:, -1]).mean()
                return {"x_mae": x_mae, "y_mae": y_mae}
            else:
                x_mae = torch.abs(predictions[:, :, :-1] - targets[:, None, :-1]).sum(-1).mean(0)
                y_mae = torch.abs(predictions[:, :, -1] - targets[:, None, -1]).mean(0)
                return {"x_mae": x_mae, "y_mae": y_mae}

        accuracy = (predictions == targets[:, None]).to(torch.float).mean()
        mae = torch.abs(predictions - targets[:, None]).to(torch.float).mean()
        return {"accuracy": accuracy, "mae": mae}

    def get_predictions(self, outputs, do_sample=False, temperature=1.0):
        """
        outputs - [batch_size, seq_len + 1, action_dim]
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
        return {
            "outputs": outputs,
            "predictions": self.get_predictions(outputs),
            "targets": batch["target_action"]
        }

    def _online_step(self, batch):
        outputs = []
        all_predictions = []
        best_prediction = []
        for query_state, problem in zip(batch["query_state"], batch["problem"]):
            results = self.run(
                query_state=query_state,
                problem=problem,
                n_steps=self.config["online_steps"],
                do_sample=self.config["do_sample"],
                temperature_function=lambda x: self.config["temperature"]
            )
            outputs.append(results["outputs"])
            all_predictions.append(results["next_states"])
            best_prediction.append(results["best_state"])
        return {
            "outputs": torch.stack(outputs),
            "all_predictions": torch.stack(all_predictions),
            "best_prediction": torch.stack(best_prediction),
            "targets": batch["target_state"]
        }

    def run(self, query_state, problem, n_steps=10, do_sample=False, temperature_function=lambda x: 1.0):
        """
        run an online inference
        """
        device = query_state.device
        state_dim = self.config["model_params"]["state_dim"]

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
        # [1, 0, state_dim]
        outputs = torch.Tensor(1, 0, state_dim).to(dtype=torch.float, device=device)

        for n_step in range(n_steps):
            # [1, state_dim]
            output = self.model(
                query_state=query_state,
                states=states,
                actions=actions,
                next_states=next_states,
                rewards=rewards
            )[:, -1, :]
            # [1]
            predicted_action = self.get_predictions(output, do_sample=do_sample, temperature=temperature_function(n_step))
            # [1, state_dim]
            predicted_state = query_state.clone()
            if predicted_action < problem.d:
                predicted_state[0][predicted_action[0]] = torch.abs(1 - predicted_state[0][predicted_action[0]])
                predicted_state[0][-1] = problem.target(predicted_state[0][:-1].cpu().detach().numpy())
            # [1, n_step, state_dim]
            states = torch.cat([states, query_state.unsqueeze(1)], dim=1)
            # [1, n_step]
            actions = torch.cat([actions, predicted_action.unsqueeze(1)], dim=1)
            # [1, n_step, state_dim]
            next_states = torch.cat([next_states, predicted_state.unsqueeze(1)], dim=1)
            # [1]
            reward = torch.tensor([0.0], device=device)
            # --------------------------------------------------------------------------------------------------
            # нечестный reward
            # if predicted_action < problem.d:
            #     if predicted_state[0][predicted_action[0][0]] == problem.info["x_min"][predicted_action[0][0]]:
            #         reward = torch.tensor([1.0], device=device)
            #     else:
            #         reward = torch.tensor([0.0], device=device)
            # else:
            #     if torch.all(predicted_state[0][:-1].cpu() == problem.info["x_min"].copy()):
            #         reward = torch.tensor([1.0], device=device)
            #     else:
            #         reward = torch.tensor([0.0], device=device)
            # --------------------------------------------------------------------------------------------------
            # [1, n_step]
            rewards = torch.cat([rewards, reward.unsqueeze(1)], dim=1)
            # [1, n_step, state_dim]
            outputs = torch.cat([outputs, output.unsqueeze(1)], dim=1)

            query_state = predicted_state

        # [1, n_step, state_dim]
        y = next_states[0, :, -1]
        best_state = next_states[0, torch.argmin(y, -1)]

        return {
            "query_state": query_state[0],
            "states": states[0],
            "actions": actions[0],
            "next_states": next_states[0],
            "rewards": rewards[0],
            "outputs": outputs[0],
            "best_state": best_state
        }


