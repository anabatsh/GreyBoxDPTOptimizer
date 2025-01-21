import torch
from torch import nn
from torch.nn import functional as F
from scipy.signal.windows import gaussian


class CE_Loss(nn.Module):
    def __init__(self):
        """
        L(p, q) = cross_entropy(predictions, targets)
        """
        super().__init__()

    def __call__(self, predictions, targets):
        """
        predictions - [batch_size, seq_len, num_actions]
        targets     - [batch_size]
        """
        # [batch_size] -> [batch_size, 1] -> [batch_size, seq_len]
        # targets = targets[:, None].repeat(1, predictions.shape[1])
        # predictions = predictions.transpose(1, 2)
        predictions = predictions[:, -1, :]
        loss = F.cross_entropy(predictions, targets)
        return loss

def int2bin(x, d, n):
    i = []
    for _ in range(d):
        i.append(x % n)
        x = x // n
    i = torch.stack(i).T.flip(-1)
    return i

class MSE_Loss(nn.Module):
    def __init__(self):
        """
        L(p, q) = mse(predictions, targets)
        """
        super().__init__()
        # self.base = torch.pow(2, torch.arange(10).flip(-1)).to(torch.float32)

    def __call__(self, outputs, targets):
        """
        predictions - [batch_size, seq_len, num_actions]
        targets     - [batch_size]
        """
        # outputs = outputs[:, -1, :]
        # # return ((outputs - int2bin(targets, 10, 2).to(outputs.dtype)) ** 2 * self.base[None].to(outputs.device)).sum()
        # return F.mse_loss(outputs, targets/1023)
        # # loss = F.mse_loss(outputs, int2bin(targets, 10, 2).to(outputs.dtype))
        # outputs = outputs[:, -1, 0]
        # outputs = F.sigmoid(outputs)
        
        outputs = outputs[:, -1, :]
        outputs = F.softmax(outputs, -1)
        targets = F.one_hot(targets, num_classes=outputs.shape[-1]).to(outputs.dtype)
        loss = F.mse_loss(outputs, targets)
        return loss

class NLLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, outputs, targets):
        """
        predictions - [batch_size, seq_len, num_actions]
        targets     - [batch_size]
        """
        outputs = outputs[:, -1, :]
        outputs = F.log_softmax(outputs, -1)
        loss = F.nll_loss(outputs, targets)
        return loss