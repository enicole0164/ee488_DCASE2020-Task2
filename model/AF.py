# FPH.py
import torch.nn as nn
import torch


# BasicHead and LeakyReLUHead are referred to https://github.com/huangswt/OneStage-SCL/blob/main/OS-SCL/model/AF.py
class BasicHead(nn.Module):
    def __init__(self):
        super(BasicHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
        )

    def forward(self, x):
        return self.head(x)


class LeakyReLUHead(nn.Module):
    def __init__(self,cfg):
        super(LeakyReLUHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.BatchNorm1d(cfg['htn']),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.BatchNorm1d(128),
        )

    def forward(self, x):
        return self.head(x)

