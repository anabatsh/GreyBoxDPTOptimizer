import torch
import torch.nn as nn
from torch.nn import functional as F
import lightning as L

try:
    from model import DPT
    from utils.schedule import cosine_annealing_with_warmup
except ImportError:
    from .model import DPT
    from .utils.schedule import cosine_annealing_with_warmup

class MSE_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, outputs, targets):
        """
        predictions - [batch_size, seq_len, num_actions]
        targets     - [batch_size]
        """
        # [batch_size, seq_len + 1, output_dim] -> [batch_size]
        outputs = outputs[:, -1, 0]
        loss = F.mse_loss(outputs, targets)
        return loss
    
class DPTSolver(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = DPT(**config["model_params"])
        self.loss = MSE_Loss()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        outputs = self._offline_step(batch)
        results = self.get_loss(**outputs)
        for key, val in results.items():
            self.log(f"train {key}", val, on_step=True, on_epoch=False)
        return results

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self._offline_step(batch)
        results = self.get_loss(**outputs)
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

    def get_loss(self, outputs, targets):
        return {"loss": self.loss(outputs, targets)}

    def _offline_step(self, batch):
        outputs = self.model(x=batch["x"], y=batch["y"])
        return {
            "outputs": outputs,
            "targets": batch["x_min"]
        }
