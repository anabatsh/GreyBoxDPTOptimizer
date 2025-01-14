import torch
from torch.nn import functional as F
from scipy.signal.windows import gaussian

device = "cuda" if torch.cuda.is_available() else "cpu"

    
class Loss():
    def __init__(self, num_classes=1024, eps=0.5, mode="dynamic"):
        """
        p - predicted distribution
        q - delta distribution
        L(p, q) = eps_i * cross_entropy(p, uniform) + (1 - eps_i) * cross_entropy(p, gaussian)
                  eps_i = 1 - i / (seq_len - 1)
        """
        width = 2 * int(0.5 * eps * num_classes) - 1
        std = width / 6
        kernel = torch.from_numpy(gaussian(width, std=std)).to(torch.float32)[None, None, :]
        one_hot_targets = torch.eye(num_classes).to(torch.float32)
        smoothed_targets = F.conv1d(one_hot_targets.unsqueeze(1), kernel, padding=kernel.shape[-1]//2).squeeze(1)
        smoothed_targets /= smoothed_targets.sum(-1)[:, None]
        self.smoothed_targets = smoothed_targets.to(device)
        self.num_classes = num_classes
        self.mode = mode
        
    def __call__(self, predictions, targets, alpha=0, reduction='mean'):
        """
        predictions - [batch_size, seq_len, num_actions]
        targets     - [batch_size, seq_len]
        """
        u_gaussian = self.smoothed_targets[targets]
        if predictions.ndim == 3:
            predictions = predictions.transpose(1, 2)
            u_gaussian = u_gaussian.transpose(1, 2)
        u = u_gaussian
        if self.mode == "dynamic":
            u_uniform = torch.ones_like(predictions).to(device) / self.num_classes
            if predictions.ndim == 3:
                # u_uniform = u_uniform.transpose(1, 2)
                alpha = 1 - torch.arange(predictions.shape[-1]).to(device) / (predictions.shape[-1] - 1)
            u = alpha * u_uniform + (1 - alpha) * u_gaussian
        self.u = u
        loss = F.cross_entropy(predictions, u, reduction=reduction)
        return loss