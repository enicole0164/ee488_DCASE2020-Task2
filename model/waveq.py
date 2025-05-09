# waveq.py
import torch.nn as nn
import torch.nn.functional as F

# TFgram and other models referred to https://github.com/huangswt/OneStage-SCL/blob/main/OS-SCL/model/waveq.py
class ConvPreWavBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvPreWavBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1, dilation=2,
                               padding=2, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.init_weight()

    def forward(self, input, pool_size=(1, 1)):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))

        x = F.max_pool1d(x, kernel_size=pool_size)

        return x


class ConvPreWavBlockqo(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvPreWavBlockqo, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1, dilation=2,
                               padding=2, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.init_weight()

    def forward(self, input, pool_size):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.adaptive_max_pool1d(x, 626)
        return x


class ConvPreWavBlockqt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvPreWavBlockqt, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1, dilation=2,
                               padding=2, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.init_weight()

    def forward(self, input, pool_size):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.adaptive_max_pool1d(x, 313)
        return x


class TFgram(nn.Module):
    def __init__(self, classes_num):
        super(TFgram, self).__init__()

        self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=5, padding=5, bias=False)
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = ConvPreWavBlock(64, 64)
        self.pre_block2 = ConvPreWavBlockqo(64, 128)
        self.pre_block3 = ConvPreWavBlockqt(128, 128)

    def forward(self, input, train):
        """
        Input: (batch_size, data_length)"""

        # TFgram
        if train:
            input = input.squeeze()
        else:
            input = input.squeeze().unsqueeze(0)
        a1 = F.relu_(self.pre_bn0(self.pre_conv0(input[:, None, :])))  # （32，64，32000 ）
        a1 = self.pre_block1(a1, pool_size=4)  # （32，64，8000）
        a1 = self.pre_block2(a1, pool_size=4)
        a1 = self.pre_block3(a1, pool_size=4)
        return a1
