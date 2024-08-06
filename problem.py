import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os


class MyNet():
    def __init__(self, d=10, n=2, seed=42):
        self.d = d
        self.n = n

        kernel_size = 5
        self.f = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size, stride=kernel_size, bias=False),
            nn.Linear(d//kernel_size, 1),
            # nn.BatchNorm()
        )
        self.c = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size, stride=kernel_size, bias=False),
            nn.Linear(d//kernel_size, 1),
            nn.Sigmoid()
        )

    def target(self, i):
        x = torch.tensor(i.copy()).to(torch.float32).reshape(-1, 1, self.d)
        y = self.f(x)[:, 0, 0].detach().numpy()
        return y[0] if len(y) == 1 else y
    
    def constr(self, i):
        x = torch.tensor(i.copy()).to(torch.float32).reshape(-1, 1, self.d)
        y = self.c(x)[:, 0, 0].detach().numpy() > 0.5
        return y[0] if len(y) == 1 else y

    def int2bin(self, x):
        i = []
        for _ in range(self.d):
            i.append(x % self.n)
            x = x // self.n
        i = np.array(i)[::-1].T
        return i
    
    def get_xaxis(self):
        t = 10
        if self.d > t:
            d = self.d - t
            x = np.arange(0, self.n ** (self.d - d))
            i = self.int2bin(x)[:, d:]
            i = np.hstack([i, np.zeros((i.shape[0], d), np.int32)])
        else:
            x = np.arange(0, self.n ** self.d)
            i = self.int2bin(x)
        return i
    
    def full_plot(self, save_dir=''):
        save_path = os.path.join(save_dir, 'problem.png')

        i = self.get_xaxis()
        y = self.target(i)
        # y[self.constr(i)] = None

        plt.figure(figsize=(8, 4), facecolor='black')
        ax = plt.axes()
        ax.set_facecolor('black')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white') 
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(colors='white', which='both')

        plt.title('Target Function', color='white')
        plt.plot(y, '-o', markersize=1)
        plt.xticks([0, len(y)-1], [fr'$[0]^{{{self.d}}}$', fr'$[{self.n-1}]^{{{self.d}}}$'])
        # plt.show()
        plt.savefig(save_path)