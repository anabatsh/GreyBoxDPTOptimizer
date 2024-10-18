import numpy as np
import torch
import torch.nn as nn


class Net():
    def __init__(self, d=10, n=2, seed=1):
        torch.manual_seed(seed)

        self.d = d
        self.n = n

        # kernel_size = 5
        self.f = lambda x: -((x - 1) ** 2).sum(-1)#nn.Sequential(
            # nn.Linear(d, 1)
            # nn.Conv1d(1, 1, kernel_size, stride=kernel_size, bias=False),
            # nn.Linear(d//kernel_size, 1),
            # nn.BatchNorm()
        # )
        # self.c = nn.Sequential(
        #     nn.Conv1d(1, 1, kernel_size, stride=kernel_size, bias=False),
        #     nn.Linear(d//kernel_size, 1),
        #     nn.Sigmoid()
        # )

    def target(self, i):
        # i = np.array(i).reshape(-1, )
        # print(i, i.ndim, i.shape)
        # x = torch.tensor(i.copy()).to(torch.float32).reshape(-1, 1, self.d)
        # y = self.f(x)[:, 0, 0].detach().numpy()
        # return y[0] if i.ndim == 1 else y
        x = torch.tensor(np.array(i)).to(torch.float32).reshape(-1, self.d)
        y = self.f(x).flatten().detach().numpy() * 10
        return y
        
    
    # def constr(self, i):
    #     i = np.array(i)
    #     x = torch.tensor(i.copy()).to(torch.float32).reshape(-1, 1, self.d)
    #     y = self.c(x)[:, 0, 0].detach().numpy() > 0.5
    #     return y[0] if len(y) == 1 else y