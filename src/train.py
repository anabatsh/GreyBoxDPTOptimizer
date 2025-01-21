import torch
import torch.nn as nn
from torch.nn import functional as F
import lightning as L

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

    def training_step(self, batch, batch_idx):
        outputs = self._offline_step(batch)
        results = self.get_loss(**outputs) | self.get_metrics(**outputs)
        for key, val in results.items():
            self.log(f"train {key}", val, on_step=True, on_epoch=False)
        return results

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if len(batch.keys()) > 2:
            outputs = self._offline_step(batch)
        else:
            outputs = self._online_step(batch)
        results = self.get_loss(**outputs) | self.get_metrics(**outputs)
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

    def get_loss(self, outputs, targets, predictions=None):
        """
        outputs - [batch_size, seq_len + 1, output_dim]
        targets - [batch_size, input_dim]
        """
        outputs = outputs[:, -1, :]
        loss = F.mse_loss(outputs, targets)
        return {"loss": loss}

    def get_predictions(self, outputs, do_sample=False, temperature=1.0):
        """
        outputs - [batch_size, seq_len + 1, output_dim]
        """
        if do_sample and temperature > 0:
            probs = F.softmax(outputs / temperature, dim=-1)
            predictions = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            # input_dim = 1, output_dim = 1
            predictions = outputs
        return predictions
    
    def get_metrics(self, outputs, targets, predictions=None):
        """
        outputs     - [batch_size, seq_len + 1, output_dim]
        predictions - [batch_size, seq_len + 1, input_dim]
        targets     - [batch_size, input_dim]
        """
        accuracy = (torch.all(predictions[:, -1] == targets, dim=-1)).to(torch.float).mean()
        mae = torch.abs(predictions[:, -1] - targets).to(torch.float).mean()
        return {"accuracy": accuracy, "mae": mae}
    
    def _offline_step(self, batch):
        outputs = self.model(x=batch["x"], y=batch["y"])
        return {
            "outputs": outputs,
            "predictions": self.get_predictions(outputs),
            "targets": batch["x_min"]
        }

    def _online_step(self, batch):
        outputs = []
        predictions = []
        for problem in batch["problem"]:
            results = self.run(
                problem=problem,
                n_steps=self.config["model_params"]["seq_len"] + 1
            )
            outputs.append(results["outputs"])
            predictions.append(results["x"])
        return {
            "outputs": torch.stack(outputs),
            "predictions": torch.stack(predictions),
            "targets": batch["x_min"]
        }

    def run(self, problem, n_steps=10, do_sample=False, temperature_function=lambda x: 1.0):
        """
        """
        device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = self.config["model_params"]["output_dim"]
        output_dim = self.config["model_params"]["output_dim"]

        # [1, 0, output_dim]
        outputs = torch.Tensor(1, 0, output_dim).to(dtype=torch.long, device=device)
        # [1, 0, input_dim]
        x = torch.Tensor(1, 0, input_dim).to(dtype=torch.long, device=device)
        # [1, 0]
        y = torch.Tensor(1, 0).to(dtype=torch.float, device=device)

        for n_step in range(n_steps):
            # [1, output_dim]
            output = self.model(x=x, y=y)[:, -1, :]
            # [1, input_dim]
            predicted_x = self.get_predictions(output, do_sample=do_sample, temperature=temperature_function(n_step))
            # [1]
            predicted_y = torch.FloatTensor(problem.target(predicted_x.detach().numpy()))#.to(dtype=torch.float)
            # [1, n_step, output_dim]
            outputs = torch.cat([outputs, output.unsqueeze(1)], dim=1)
            # [1, n_step, input_dim]
            x = torch.cat([x, predicted_x.unsqueeze(1)], dim=1)
            # [1, n_step]
            y = torch.cat([y, predicted_y.unsqueeze(1)], dim=1)

        return {
            "outputs": outputs[0],
            "x": x[0],
            "y": y[0]
        }


