# waveq.py
import torch.nn as nn
import torch.nn.functional as F
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

#---------------wavenet----------------------------------------------------

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv1 = self.causal_conv(self.in_channels, self.out_channels, self.kernel_size, self.dilation)
        self.padding = self.conv1.padding[0]

    def causal_conv(self, in_channels, out_channels, kernel_size, dilation):
        pad = (kernel_size - 1) * dilation
        return nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation)

    def forward(self, x):
        x = self.conv1(x)
        x = x[:, :, :-self.padding]
        return x


class ResidualBlock(nn.Module):
    def __init__(self, n_channel, n_mul, kernel_size, dilation_rate):
        super(ResidualBlock, self).__init__()
        self.n_channel = n_channel
        self.n_mul = n_mul
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_filter = self.n_channel * self.n_mul

        self.sigmoid_group_norm = nn.GroupNorm(1, self.n_filter)
        self.sigmoid_conv = CausalConv1d(self.n_filter, self.n_filter, self.kernel_size, self.dilation_rate)
        self.tanh_group_norm = nn.GroupNorm(1, self.n_filter)
        self.tanh_conv = CausalConv1d(self.n_filter, self.n_filter, self.kernel_size, self.dilation_rate)

        self.skip_group_norm = nn.GroupNorm(1, self.n_filter).to(device)
        self.skip_conv = nn.Conv1d(self.n_filter, self.n_channel, 1)
        self.residual_group_norm = nn.GroupNorm(1, self.n_filter)
        self.residual_conv = nn.Conv1d(self.n_filter, self.n_filter, 1)

    def forward(self, x):
        x1 = self.sigmoid_group_norm(x)
        x1 = self.sigmoid_conv(x1)
        x2 = self.tanh_group_norm(x)
        x2 = self.tanh_conv(x2)
        x1 = nn.Sigmoid()(x1)
        x2 = nn.Tanh()(x2)
        x = x1 * x2
        x1 = self.skip_group_norm(x)
        skip = self.skip_conv(x1)
        x2 = self.residual_group_norm(x)
        residual = self.residual_conv(x2)
        return skip, residual


class WaveNet(nn.Module):
    def __init__(self, n_channel, n_mul, kernel_size):
        super(WaveNet, self).__init__()
        self.n_channel = n_channel
        self.n_mul = n_mul
        self.kernel_size = kernel_size

        self.n_filter = self.n_channel * self.n_mul
        self.group_norm1 = nn.GroupNorm(1, self.n_channel)
        self.conv1 = nn.Conv1d(self.n_channel, self.n_filter, 1)

        self.block1 = ResidualBlock(self.n_channel, self.n_mul, self.kernel_size, 1)
        self.block2 = ResidualBlock(self.n_channel, self.n_mul, self.kernel_size, 2)
        self.block3 = ResidualBlock(self.n_channel, self.n_mul, self.kernel_size, 4)

        self.relu1 = nn.ReLU()
        self.group_norm2 = nn.GroupNorm(1, self.n_channel)
        self.conv2 = nn.Conv1d(self.n_channel, self.n_channel, 1)
        self.relu2 = nn.ReLU()
        self.group_norm3 = nn.GroupNorm(1, self.n_channel)
        self.conv3 = nn.Conv1d(self.n_channel, self.n_channel, 1)

    def forward(self, x):
        x = self.group_norm1(x)
        x = self.conv1(x)
        skip1, x = self.block1(x)
        skip2, x = self.block2(x)
        skip3, x = self.block3(x)
        skip = skip1 + skip2 + skip3
        x = self.relu1(skip)
        x = self.group_norm2(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.group_norm3(x)
        x = self.conv3(x)

        skip1 = self.conv3(self.group_norm3(self.relu2(self.conv2(self.group_norm2(self.relu1(skip1))))))
        skip2 = self.conv3(self.group_norm3(self.relu2(self.conv2(self.group_norm2(self.relu1(skip2))))))
        skip3 = self.conv3(self.group_norm3(self.relu2(self.conv2(self.group_norm2(self.relu1(skip3))))))
        output = x[:, :, self.get_receptive_field() - 1:-1]

        skip1 = skip1[:, :, self.get_receptive_field() - 1:-1]
        skip2 = skip2[:, :, self.get_receptive_field() - 1:-1]
        skip3 = skip3[:, :, self.get_receptive_field() - 1:-1]
        return output, skip1, skip2, skip3

    def get_receptive_field(self):
        receptive_field = 1
        for _ in range(3):
            receptive_field = receptive_field * 2 + self.kernel_size - 2
        return receptive_field

class WaveNet_jaeryeong(nn.Module):
    def __init__(self, n_channel, n_mul, kernel_size):
        super(WaveNet_jaeryeong, self).__init__()
        self.wavegram = nn.Conv1d(1, 128, kernel_size=1024, stride=512, padding=1024 // 2, bias=False)
        self.causal_conv = CausalConv1d(128, 512, kernel_size=2, dilation=1)
        
        self.n_channel = n_channel
        self.n_mul = n_mul
        self.kernel_size = kernel_size

        self.n_filter = self.n_channel * self.n_mul
        self.group_norm1 = nn.GroupNorm(1, self.n_channel)
        self.conv1 = nn.Conv1d(self.n_channel, self.n_filter, 1)

        self.block1_1 = ResidualBlock(512, self.n_mul, self.kernel_size, 1)
        self.block1_2 = ResidualBlock(512, self.n_mul, self.kernel_size, 1)
        self.block1_3 = ResidualBlock(512, self.n_mul, self.kernel_size, 1)
        self.block2_1 = ResidualBlock(512, self.n_mul, self.kernel_size, 2)
        self.block2_2 = ResidualBlock(512, self.n_mul, self.kernel_size, 2)
        self.block2_3 = ResidualBlock(512, self.n_mul, self.kernel_size, 2)
        self.block3_1 = ResidualBlock(512, self.n_mul, self.kernel_size, 4)
        self.block3_2 = ResidualBlock(512, self.n_mul, self.kernel_size, 4)
        self.block3_3 = ResidualBlock(512, self.n_mul, self.kernel_size, 4)
        self.block4_1 = ResidualBlock(512, self.n_mul, self.kernel_size, 8)
        self.block4_2 = ResidualBlock(512, self.n_mul, self.kernel_size, 8)
        self.block4_3 = ResidualBlock(512, self.n_mul, self.kernel_size, 8)

        self.relu1 = nn.ReLU()
        self.group_norm2 = nn.GroupNorm(1, self.n_channel * 4)
        self.conv2 = nn.Conv1d(self.n_channel * 4, self.n_channel * 4, 1)
        self.relu2 = nn.ReLU()
        self.group_norm3 = nn.GroupNorm(1, self.n_channel * 4)
        self.conv3 = nn.Conv1d(self.n_channel * 4, self.n_channel, 1)

    def forward(self, x):
        x = self.wavegram(x)
        x = self.group_norm1(x)
        x = self.causal_conv(x)

        skip1_1, x = self.block1_1(x)
        skip1_2, x = self.block1_2(x)
        skip1_3, x = self.block1_3(x)
        skip2_1, x = self.block2_1(x)
        skip2_2, x = self.block2_2(x)
        skip2_3, x = self.block2_3(x)
        skip3_1, x = self.block3_1(x)
        skip3_2, x = self.block3_2(x)
        skip3_3, x = self.block3_3(x)
        skip4_1, x = self.block4_1(x)
        skip4_2, x = self.block4_2(x)
        skip4_3, x = self.block4_3(x)

        skip = skip1_1 + skip1_2 + skip1_3 \
                + skip2_1 + skip2_2 + skip2_3 \
                + skip3_1 + skip3_2 + skip3_3 \
                + skip4_1 + skip4_2 + skip4_3
        
        x = self.relu1(skip)
        x = self.group_norm2(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.group_norm3(x)
        x = self.conv3(x)

        return x

    def get_receptive_field(self):
        receptive_field = 1
        for _ in range(3):
            receptive_field = receptive_field * 2 + self.kernel_size - 2
        return receptive_field


class WaveNet_team2(nn.Module):
    def __init__(self, n_channel, n_mul, kernel_size):
        super(WaveNet_team2, self).__init__()
        self.wavegram = nn.Conv1d(1, 128, kernel_size=1024, stride=512, padding=1024 // 2, bias=False)
        self.causal_conv = CausalConv1d(128, 512, kernel_size=2, dilation=1)
        
        self.n_channel = n_channel
        self.n_mul = n_mul
        self.kernel_size = kernel_size

        self.n_filter = self.n_channel * self.n_mul
        self.group_norm1 = nn.GroupNorm(1, self.n_channel)
        self.conv1 = nn.Conv1d(self.n_channel, self.n_filter, 1)

        self.block1 = ResidualBlock(512, self.n_mul, self.kernel_size, 1)
        self.block2 = ResidualBlock(512, self.n_mul, self.kernel_size, 2)
        self.block3 = ResidualBlock(512, self.n_mul, self.kernel_size, 4)
        self.block4 = ResidualBlock(512, self.n_mul, self.kernel_size, 8)

        self.relu1 = nn.ReLU()
        self.group_norm2 = nn.GroupNorm(1, self.n_channel * 4)
        self.conv2 = nn.Conv1d(self.n_channel * 4, self.n_channel * 4, 1)
        self.relu2 = nn.ReLU()
        self.group_norm3 = nn.GroupNorm(1, self.n_channel * 4)
        self.conv3 = nn.Conv1d(self.n_channel * 4, self.n_channel, 1)

    def forward(self, x):
        x = self.wavegram(x)
        x = self.group_norm1(x)
        x = self.causal_conv(x)

        skip1, x = self.block1(x)
        skip2, x = self.block2(x)
        skip3, x = self.block3(x)
        skip4, x = self.block4(x)

        skip = skip1 + skip2 + skip3 + skip4
        
        x = self.relu1(skip)
        x = self.group_norm2(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.group_norm3(x)
        x = self.conv3(x)

        return x

    def get_receptive_field(self):
        receptive_field = 1
        for _ in range(3):
            receptive_field = receptive_field * 2 + self.kernel_size - 2
        return receptive_field
