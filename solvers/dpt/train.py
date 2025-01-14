import torch
from torch.nn import functional as F
import lightning as L
from scipy.signal.windows import gaussian
from torch.distributions.normal import Normal
try:
    from model import DPT_K2D
    from utils.loss import Loss
    from utils.schedule import cosine_annealing_with_warmup
except ImportError:
    from .model import DPT_K2D
    from .utils.loss import Loss
    from .utils.schedule import cosine_annealing_with_warmup

device = "cuda" if torch.cuda.is_available() else "cpu"


class DPTSolver(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = DPT_K2D(**config["model_params"]).to(device)
        self.loss = Loss(
            num_classes=self.config["model_params"]["num_actions"], 
            eps=self.config["label_smoothing"],
            # mode=self.config["loss"]
        )
        self.save_hyperparameters()

    def get_predictions(self, logits, do_sample=False, temperature=1.0):
        if do_sample and temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            predictions = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            predictions = torch.argmax(logits, dim=-1)
        return predictions

    def get_loss(self, logits, targets):
        if logits.ndim == 3 and targets.ndim == 1:
            # [batch_size] -> [batch_size, 1] -> [batch_size, seq_len]
            targets = targets[:, None].repeat(1, logits.shape[1])
        loss = self.loss(logits, targets)
        return {"loss": loss}
    
    def get_metrics(self, logits, targets, predictions=None, batch=None):
        with torch.no_grad():
            if predictions == None:
                predictions = self.get_predictions(logits)
            targets = targets[:, None] if (predictions.ndim == 2 and targets.ndim == 1) else targets
            # accuracy = (action == target).to(torch.float).mean()
            accuracy = ((predictions == targets).sum(-1) >= 1).to(torch.float).mean()
            mse = torch.stack([
                torch.sqrt(((problem.target(prediction) - problem.target(target)) ** 2).sum(-1))
                for prediction, target, problem in zip(predictions, targets, batch['problem'])
            ]).mean()
        return {"accuracy": accuracy, "mse": mse}

    def _offline_step(self, batch, batch_idx):
        logits = self.model(
            x=batch["x"].to(device),
            y=batch["y"].to(device),
        )
        targets = batch["x_min"].to(device)
        return self.get_loss(logits, targets) | self.get_metrics(logits, targets, batch=batch)

    def _online_step(self, batch, batch_idx):
        offline_batch = {"x": [], "y": [], "logits": []}
        for problem in batch["problem"]:
            results = self.run(
                problem=problem,
                n_steps=self.config["model_params"]["seq_len"]
            )
            for key, val in results.items():
                offline_batch[key].append(val)
        for key, val in offline_batch.items():
            offline_batch[key] = torch.stack(val)
        
        logits = offline_batch.pop("logits")
        targets = batch["x_min"].to(device)
        # offline_batch |= batch
        return self.get_loss(logits, targets) | self.get_metrics(logits, targets, predictions=offline_batch["x"], batch=batch)

    def training_step(self, batch, batch_idx):
        results = self._offline_step(batch, batch_idx)
        for key, val in results.items():
            self.log(f"train {key}", val, on_step=True, on_epoch=False)
        return results

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if "x" in batch and "y" in batch:
            return self._offline_step(batch, batch_idx)
        return self._online_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # if dataloader_idx == 0 or self.current_epoch % 10 == 0:
        results = self.test_step(batch, batch_idx, dataloader_idx)
        for key, val in results.items():
            self.log(f"val {key}", val, on_step=False, on_epoch=True)
        return results

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

    def run(self, problem, n_steps=10, do_sample=False, temperature_function=lambda x: 1.0):
        """
        """
        num_actions = self.config["model_params"]["num_actions"]

        # [1, 0, num_actions]
        logits = torch.Tensor(1, 0, num_actions).to(dtype=torch.long, device=device)
        # [1, 0]
        x = torch.Tensor(1, 0).to(dtype=torch.long, device=device)
        # [1, 0]
        y = torch.Tensor(1, 0).to(dtype=torch.float, device=device)

        for n_step in range(n_steps):

            x_ = x.clone()
            y_ = y.clone()
            if self.config["ordered"]:
                sort_indexes = torch.flip(torch.argsort(y_[0]), (0,))
                x_[0] = x[0][sort_indexes]
                y_[0] = y[0][sort_indexes]

            # [1, num_actions]
            predicted_logits = self.model(
                x=x_,
                y=y_
            )#[:, -1, :]
            # [1]
            predicted_x = self.get_predictions(predicted_logits, do_sample=do_sample, temperature=temperature_function(n_step))
            # [1]
            predicted_y = problem.target(predicted_x).to(dtype=torch.float)#, device=device)
            # [1, n_step, num_actions]
            logits = torch.cat([logits, predicted_logits.unsqueeze(1)], dim=1)
            # [1, n_step]
            x = torch.cat([x, predicted_x.unsqueeze(1)], dim=1)
            # [1, n_step]
            y = torch.cat([y, predicted_y.unsqueeze(1)], dim=1)

        return {
            "logits": logits[0],
            "x": x[0],
            "y": y[0]
        }
