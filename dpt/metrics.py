import torch
from torch import nn


class PointMetrics(nn.Module):
    def __init__(self):    
        super().__init__()

    def forward(self, targets, predictions, *args, **kwargs):
        """
        predictions - [batch_size, seq_len + 1, action_dim]
        targets     - [batch_size, action_dim]
        """
        targets = targets.long()
        accuracy = (predictions == targets[:, None]).float()
        mae = torch.abs(predictions - targets[:, None]).float()
        return {
            "accuracy": accuracy.mean(), 
            # "accuracy_last": accuracy[:, -1].mean(),
            "x_mae": mae.mean(),
            # "x_mae_last": mae[:, -1].mean(),
        }


class BitflipMetrics(nn.Module):
    def __init__(self):    
        super().__init__()

    def forward(self, targets, predictions, *args, **kwargs):
        """
        predictions - [batch_size, seq_len + 1, action_dim]
        targets     - [batch_size, action_dim]
        """
        targets = targets.unsqueeze(1)
        accuracy = torch.any((predictions * targets), dim=-1).float()
        predictions = torch.argmax(predictions, dim=-1)

        d = targets.shape[-1]
        indexes = torch.arange(1, d + 1, device=targets.device) * targets - 1
        indexes[indexes < 0] = 3 * d # imax
        dists = (indexes - predictions[..., None]).abs()
        mae = dists.min(dim=-1).values.float()

        return {
            "accuracy": accuracy.mean(), 
            # "accuracy_last": accuracy[:, -1].mean(),
            "x_mae": mae.mean(),
            # "x_mae_last": mae[:, -1].mean(),
        }