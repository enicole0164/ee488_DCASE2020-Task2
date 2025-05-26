"""
modification made on the basis of link:https://github.com/liuyoude/STgram-MFN
"""
import math
from torch import nn
import torch
from losses import ArcMarginProduct
from model.waveq import TFgram

import torch.nn.functional as F
from model.AF import BasicHead, LeakyReLUHead


class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        #
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),

            # pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


#https://dcase.community/documents/challenge2022/technical_reports/DCASE2022_Liu_8_t2.pdf
Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 128, 2, 2],
    [4, 128, 2, 2],
    [4, 128, 2, 2],
]


class MobileFaceNet(nn.Module):
    def __init__(self,
                 num_class,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(MobileFaceNet, self).__init__()

        self.conv1 = ConvBlock(3, 64, 3, 2, 1)

        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(bottleneck_setting[-1][1], 512, 1, 1, 0)
        # 20(10), 4(2), 8(4)
        self.linear7 = ConvBlock(512, 512, (8, 20), 1, 0, dw=True, linear=True)
        
        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)

        self.fc_out = nn.Linear(128, num_class)
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        feature = x.view(x.size(0), -1)
        out = self.fc_out(feature)
        return out, feature


class TgramNet(nn.Module):
    def __init__(self, num_layer=3, mel_bins=128, win_len=1024, hop_len=512):
        super(TgramNet, self).__init__()
        # if "center=True" of stft, padding = win_len / 2
        self.conv_extrctor = nn.Conv1d(1, mel_bins, win_len, hop_len, win_len // 2, bias=False)
        self.conv_encoder = nn.Sequential(
            *[nn.Sequential(
                # 313(10) , 63(2), 126(4)
                nn.LayerNorm(313),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(mel_bins, mel_bins, 3, 1, 1, bias=False),
            ) for _ in range(num_layer)])

    def forward(self, x):
        out = self.conv_extrctor(x)
        out = self.conv_encoder(out)
        return out
    

class TASTgramMFN(nn.Module):
    def __init__(self, num_classes, mode,
                 c_dim=128,
                 win_len=1024,
                 hop_len=512,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting,
                 use_arcface=True, m=0.7, s=30, sub=1
                 ):
        super().__init__()
        
        self.arcface = ArcMarginProduct(in_features=c_dim, out_features=num_classes,
                                        m=m, s=s, sub=sub) if use_arcface else use_arcface
        self.tgramnet = TgramNet(mel_bins=c_dim, win_len=win_len, hop_len=hop_len)
        self.mobilefacenet = MobileFaceNet(num_class=num_classes,
                                           bottleneck_setting=bottleneck_setting)
        self.mode = mode
        
        if mode not in ['arcface', 'arcmix', 'noisy_arcmix']:
            raise ValueError('Choose one of [arcface, arcmix, noisy_arcmix]')
        
        self.temporal_attention = Temporal_Attention(feature_dim=c_dim)
        
    def get_tgram(self, x_wav):
        return self.tgramnet(x_wav)

    def forward(self, x_wav, x_mel, label, train=True):
        x_t = self.tgramnet(x_wav).unsqueeze(1)
        
        x_mel_temp_att = self.temporal_attention(x_mel).unsqueeze(1)
       
        x = torch.cat((x_t, x_mel, x_mel_temp_att), dim=1)
        
        out, feature = self.mobilefacenet(x)
        
        if self.mode == 'arcmix':
            if train:
                out = self.arcface(feature, label[0])
                out_shuffled = self.arcface(feature, label[1])
                return out, out_shuffled, feature
            else:
                out = self.arcface(feature, label)
                return out, feature
        
        else:
            out = self.arcface(feature, label)
            return out, feature
        
        
class Temporal_Attention(nn.Module):
  def __init__(self, feature_dim=128):
    super().__init__()
    
    self.feature_dim = feature_dim
    self.max_pool = nn.AdaptiveMaxPool1d(1)
    self.avg_pool = nn.AdaptiveAvgPool1d(1)
    self.sigmoid = nn.Sigmoid()
    
  def forward(self, x):
    # x: (B, 1, 128, 313)
    x = x.squeeze(1)
    
    x = x.transpose(1,2) # (B, 313, 128)

    x1 = self.max_pool(x) # (B, 313, 1)
    x2 = self.avg_pool(x) # (B, 313, 1)
    
    feats = x1 + x2
    
    feats = feats.repeat(1, 1, self.feature_dim)
    
    refined_feats = self.sigmoid(feats).transpose(1,2) * x.transpose(1,2)
    return refined_feats

class SCLTFSTgramMFN(nn.Module):

    def __init__(self, num_classes, mode, cfg,
                 c_dim=128,
                 win_len=1024,
                 hop_len=512,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting,
                 margin='arcface', m=0.7, s=30, sub=1, nsc=32,
                 ):
        super().__init__()

        self.margin = margin
        self.cfg = cfg
        print(m)

        self.arcface = ArcMarginProduct(in_features=c_dim, out_features=num_classes,
                                        m=m, s=s, sub=sub)

        self.tgramnet = TgramNet(mel_bins=c_dim, win_len=win_len, hop_len=hop_len)
        self.mobilefacenet = MobileFaceNet(num_class=num_classes,
                                           bottleneck_setting=bottleneck_setting,
                                           cfg=cfg)
        self.mode = mode

        self.TFgramNet = TFgram(classes_num=41)
        head_type = cfg["ht"]
        if head_type == 'basic':
            self.head = BasicHead()
        elif head_type == 'leaky_relu':
            self.head = LeakyReLUHead(cfg)

        if mode not in ['arcface']:
            raise ValueError('Choose arcface mode')

    def get_tgram(self, x_wav):
        return self.tgramnet(x_wav)

    def forward(self, x_wav, x_mel, label, train=True):
        x_t = self.tgramnet(x_wav).unsqueeze(1)  # (32,1,128,313)

        x_tf = self.TFgramNet(x_wav, train).unsqueeze(1)

        if self.cfg['fussion'] == 1:
            # f=1
            x = torch.cat((x_mel, x_t, x_tf), dim=1)

        elif self.cfg['fussion'] == 2:
            # f=2
            x = x_mel

        out, feature = self.mobilefacenet(x)

        feature = F.normalize(self.head(feature), dim=1)
        # feature = F.normalize(feature, dim=1)                 #no head
        out = self.arcface(feature, label, training=train)
        return out, feature
    

class TFgramNet(nn.Module):
    """
    Time-Frequency Gram Network based on OS-SCL paper
    Extracts joint time-frequency features from mel spectrograms
    """
    def __init__(self, mel_bins=128, time_frames=313, num_layers=3):
        super(TFgramNet, self).__init__()
        
        self.mel_bins = mel_bins
        self.time_frames = time_frames
        
        # Multi-scale time-frequency feature extraction
        self.tf_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1 if i == 0 else 32, 32, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ) for i in range(num_layers)
        ])
        
        # Time-frequency attention mechanisms
        self.temporal_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),  # Pool across frequency
            nn.Conv2d(32, 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.spectral_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),  # Pool across time
            nn.Conv2d(32, 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Joint time-frequency attention
        self.joint_attention = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Feature compression to match original dimensions
        self.feature_compress = nn.Conv2d(32, 1, kernel_size=1)
        
        # Ensure output dimensions match input
        self.adaptive_pool = nn.AdaptiveAvgPool2d((mel_bins, time_frames))
        
    def forward(self, x_mel):
        """
        Args:
            x_mel: Mel spectrogram [B, 1, mel_bins, time_frames]
        Returns:
            tf_features: Time-frequency features [B, mel_bins, time_frames]
        """
        x = x_mel
        
        # Multi-layer feature extraction with residual connections
        for i, layer in enumerate(self.tf_layers):
            residual = x if i == 0 else x_prev
            x = layer(x)
            if i > 0 and x.shape == residual.shape:
                x = x + residual
            x_prev = x
        
        # Apply multi-scale attention
        temp_att = self.temporal_attention(x)  # [B, 1, 1, time_frames]
        spec_att = self.spectral_attention(x)  # [B, 1, mel_bins, 1]
        joint_att = self.joint_attention(x)    # [B, 1, mel_bins, time_frames]
        
        # Combine attention mechanisms
        x_temp = x * temp_att
        x_spec = x * spec_att
        x_joint = x * joint_att
        
        # Aggregate all attention-enhanced features
        x_enhanced = x_temp + x_spec + x_joint
        
        # Compress features and ensure correct dimensions
        x_compressed = self.feature_compress(x_enhanced)
        x_output = self.adaptive_pool(x_compressed)
        
        return x_output.squeeze(1)  # [B, mel_bins, time_frames]


class UltraLight_TFgramNet(nn.Module):
    """
    Minimal memory version of TFgramNet
    Reduces channels and layers significantly
    """
    def __init__(self, mel_bins=128, time_frames=313):
        super(UltraLight_TFgramNet, self).__init__()
        
        # Single lightweight conv layer (instead of 3 heavy layers)
        self.tf_conv = nn.Conv2d(1, 8, kernel_size=(3, 3), padding=1)  # Only 8 channels
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        
        # Simplified attention (minimal channels)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global pooling
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Direct compression
        self.compress = nn.Conv2d(8, 1, kernel_size=1)
        
    def forward(self, x_mel):
        # Minimal processing
        x = self.tf_conv(x_mel)  # [B, 8, 128, 313]
        x = self.bn(x)
        x = self.relu(x)
        
        # Simple global attention
        att = self.attention(x)  # [B, 1, 1, 1]
        x = x * att  # Broadcast attention
        
        # Compress to output
        x = self.compress(x)  # [B, 1, 128, 313]
        
        return x.squeeze(1)  # [B, 128, 313]

"""
class TFSTgramMFN(nn.Module):
    def __init__(self, num_classes, mode,
                 c_dim=128,
                 win_len=1024,
                 hop_len=512,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting,
                 use_arcface=True, m=0.7, s=30, sub=1,
                 memory_efficient=True  # NEW FLAG
                 ):
        super().__init__()
        
        self.arcface = ArcMarginProduct(in_features=c_dim, out_features=num_classes,
                                        m=m, s=s, sub=sub) if use_arcface else use_arcface
        self.tgramnet = TgramNet(mel_bins=c_dim, win_len=win_len, hop_len=hop_len)
        self.mobilefacenet = MobileFaceNet(num_class=num_classes,
                                           bottleneck_setting=bottleneck_setting)
        self.mode = mode
        
        if mode not in ['arcface', 'arcmix', 'noisy_arcmix']:
            raise ValueError('Choose one of [arcface, arcmix, noisy_arcmix]')
        
        # Choose TFgram based on memory requirements
        if memory_efficient:
            self.tfgram_net = UltraLight_TFgramNet(mel_bins=c_dim, time_frames=313)
            print("Using Ultra-Lightweight TFgramNet for extreme memory optimization")
        else:
            self.tfgram_net = TFgramNet(mel_bins=c_dim, time_frames=313)
            print("Using full TFgramNet")
        
    def get_tgram(self, x_wav):
        return self.tgramnet(x_wav)

    def forward(self, x_wav, x_mel, label, train=True):
        x_t = self.tgramnet(x_wav).unsqueeze(1)
        x_tf = self.tfgram_net(x_mel).unsqueeze(1)
        x = torch.cat((x_t, x_mel, x_tf), dim=1)
        
        out, feature = self.mobilefacenet(x)
        
        if self.mode == 'arcmix':
            if train:
                out = self.arcface(feature, label[0])
                out_shuffled = self.arcface(feature, label[1])
                return out, out_shuffled, feature
            else:
                out = self.arcface(feature, label)
                return out, feature
        else:
            out = self.arcface(feature, label)
            return out, feature
"""
class ConvPreWavBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvPreWavBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        self.init_weight()
    
    def init_weight(self):
        """
        Initialize weights following ArcMarginProduct pattern
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Use Xavier uniform initialization like ArcMarginProduct
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                # Initialize BatchNorm parameters
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x, pool_size=None):
        residual = self.shortcut(x)
        
        out = F.relu_(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu_(out)
        
        if pool_size is not None:
            out = F.avg_pool1d(out, kernel_size=pool_size, stride=pool_size)
        
        return out


class ConvPreWavBlockqo(nn.Module):
    """ConvPreWavBlock with channel change (qo = quantity out?)"""
    def __init__(self, in_channels, out_channels):
        super(ConvPreWavBlockqo, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection with channel adjustment
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        
        self.init_weight()
    
    def init_weight(self):
        """
        Initialize weights following ArcMarginProduct pattern
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Use Xavier uniform initialization like ArcMarginProduct
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                # Initialize BatchNorm parameters
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x, pool_size=None):
        residual = self.shortcut(x)
        
        out = F.relu_(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu_(out)
        
        if pool_size is not None:
            out = F.avg_pool1d(out, kernel_size=pool_size, stride=pool_size)
        
        return out


class ConvPreWavBlockqt(nn.Module):
    """ConvPreWavBlock variant (qt = ?)"""
    def __init__(self, in_channels, out_channels):
        super(ConvPreWavBlockqt, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        self.init_weight()
    
    def init_weight(self):
        """
        Initialize weights following ArcMarginProduct pattern
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Use Xavier uniform initialization like ArcMarginProduct
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                # Initialize BatchNorm parameters
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x, pool_size=None):
        residual = self.shortcut(x)
        
        out = F.relu_(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu_(out)
        
        if pool_size is not None:
            out = F.avg_pool1d(out, kernel_size=pool_size, stride=pool_size)
        
        return out


# Complete Original TFgram implementation with updated blocks
class TFgram(nn.Module):
    def __init__(self, classes_num):
        super(TFgram, self).__init__()

        self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=5, padding=5, bias=False)
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = ConvPreWavBlock(64, 64)
        self.pre_block2 = ConvPreWavBlockqo(64, 128)
        self.pre_block3 = ConvPreWavBlockqt(128, 128)
        
        # Initialize the initial conv and bn following the same pattern
        self.init_weight()
    
    def init_weight(self):
        """
        Initialize weights for the initial layers following ArcMarginProduct pattern
        """
        # Initialize the pre_conv0 and pre_bn0
        nn.init.xavier_uniform_(self.pre_conv0.weight)
        nn.init.constant_(self.pre_bn0.weight, 1.0)
        nn.init.constant_(self.pre_bn0.bias, 0.0)

    def forward(self, input, train):
        """
        Input: Raw audio tensor - can be various shapes
        Expected: [B, audio_length] or [B, 1, audio_length]
        """
        # Debug: Print input shape to understand what we're getting
        # print(f"TFgram input shape: {input.shape}")
        
        # Handle different input shapes
        if len(input.shape) == 4:
            # If input is [B, C, H, W] (like mel spectrogram), this is wrong input
            raise ValueError(f"TFgram expects raw audio input [B, audio_length], but got shape {input.shape}. "
                           "Make sure you're passing x_wav (raw audio) not x_mel (mel spectrogram) to TFgram.")
        
        elif len(input.shape) == 3:
            # If input is [B, 1, audio_length], squeeze the middle dimension
            if input.shape[1] == 1:
                input = input.squeeze(1)  # [B, audio_length]
            else:
                # If input is [B, channels, audio_length] where channels > 1
                input = input[:, 0, :]  # Take first channel: [B, audio_length]
        
        elif len(input.shape) == 2:
            # Input is already [B, audio_length] - perfect
            pass
        
        elif len(input.shape) == 1:
            # Input is [audio_length] - add batch dimension
            input = input.unsqueeze(0)  # [1, audio_length]
        
        else:
            raise ValueError(f"Unexpected input shape: {input.shape}")
        
        # Now input should be [B, audio_length]
        # Add channel dimension for Conv1d: [B, audio_length] -> [B, 1, audio_length]
        if len(input.shape) == 2:
            input = input.unsqueeze(1)  # [B, 1, audio_length]
        
        # Apply TFgram processing
        a1 = F.relu_(self.pre_bn0(self.pre_conv0(input)))  # [B, 64, reduced_length]
        a1 = self.pre_block1(a1, pool_size=4)  # [B, 64, further_reduced]
        a1 = self.pre_block2(a1, pool_size=4)  # [B, 128, further_reduced]
        a1 = self.pre_block3(a1, pool_size=4)  # [B, 128, final_length]
        
        return a1


class TFSTgramMFN(nn.Module):
    """
    TFSTgram-MFN using Original TFgram with fixed input handling
    """
    
    def __init__(self, num_classes, mode,
                 c_dim=128,
                 win_len=1024,
                 hop_len=512,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting,
                 use_arcface=True, m=0.7, s=30, sub=1
                 ):
        super().__init__()
        
        self.arcface = ArcMarginProduct(in_features=c_dim, out_features=num_classes,
                                        m=m, s=s, sub=sub) if use_arcface else use_arcface
        
        self.tgramnet = TgramNet(mel_bins=c_dim, win_len=win_len, hop_len=hop_len)
        self.mobilefacenet = MobileFaceNet(num_class=num_classes,
                                           bottleneck_setting=bottleneck_setting)
        
        # Use complete Original TFgram implementation
        self.tfgram = TFgram(classes_num=num_classes)
        
        # Adaptive pooling to resize TFgram output to match mel spectrogram
        self.tfgram_adapter = nn.AdaptiveAvgPool2d((128, 313))
        
        self.mode = mode
        
        if mode not in ['arcface', 'arcmix', 'noisy_arcmix']:
            raise ValueError('Choose one of [arcface, arcmix, noisy_arcmix]')
        
        print("Using Original TFgram with fixed input handling")
        
    def get_tgram(self, x_wav):
        return self.tgramnet(x_wav)
    
    def get_tfgram(self, x_wav, train=True):
        return self.tfgram(x_wav, train)

    def forward(self, x_wav, x_mel, label, train=True):
        # Debug: Check input shapes
        # print(f"x_wav shape: {x_wav.shape}")  # Should be [B, 1, audio_length]
        # print(f"x_mel shape: {x_mel.shape}")  # Should be [B, 1, 128, 313]
        
        # Extract Tgram features from raw audio
        x_t = self.tgramnet(x_wav).unsqueeze(1)  # [B, 1, 128, 313]
        
        # Extract TFgram features from raw audio (NOT mel spectrogram)
        # Make sure x_wav is in correct format for TFgram
        if len(x_wav.shape) == 4:
            # If x_wav somehow became 4D, this is wrong
            raise ValueError(f"x_wav should be raw audio [B, 1, audio_length], got {x_wav.shape}")
        
        x_tf_raw = self.tfgram(x_wav, train)  # [B, 128, ~500]
        
        # Resize TFgram output to match mel spectrogram dimensions
        x_tf = self.tfgram_adapter(x_tf_raw.unsqueeze(1))  # [B, 1, 128, 313]
        
        # Concatenate all features
        x = torch.cat((x_t, x_mel, x_tf), dim=1)  # [B, 3, 128, 313]
        
        out, feature = self.mobilefacenet(x)
        
        if self.mode == 'arcmix':
            if train:
                out = self.arcface(feature, label[0])
                out_shuffled = self.arcface(feature, label[1])
                return out, out_shuffled, feature
            else:
                out = self.arcface(feature, label)
                return out, feature
        else:
            out = self.arcface(feature, label)
            return out, feature


# Alternative: Simplified TFgram if the above still has issues
class SimplifiedTFgram(nn.Module):
    """
    Simplified version of TFgram with more robust input handling
    """
    def __init__(self, classes_num):
        super(SimplifiedTFgram, self).__init__()

        self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=5, padding=5, bias=False)
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = ConvPreWavBlock(64, 64)
        self.pre_block2 = ConvPreWavBlockqo(64, 128)
        self.pre_block3 = ConvPreWavBlockqt(128, 128)
        
        self.init_weight()
    
    def init_weight(self):
        nn.init.xavier_uniform_(self.pre_conv0.weight)
        nn.init.constant_(self.pre_bn0.weight, 1.0)
        nn.init.constant_(self.pre_bn0.bias, 0.0)

    def forward(self, input, train):
        """
        Simplified forward with better error handling
        """
        # Convert input to correct shape: [B, 1, audio_length]
        if len(input.shape) == 3 and input.shape[1] == 1:
            # Already correct: [B, 1, audio_length]
            processed_input = input
        elif len(input.shape) == 2:
            # [B, audio_length] -> [B, 1, audio_length]
            processed_input = input.unsqueeze(1)
        else:
            raise ValueError(f"TFgram input must be [B, audio_length] or [B, 1, audio_length], got {input.shape}")
        
        # Process through network
        a1 = F.relu_(self.pre_bn0(self.pre_conv0(processed_input)))
        a1 = self.pre_block1(a1, pool_size=4)
        a1 = self.pre_block2(a1, pool_size=4)
        a1 = self.pre_block3(a1, pool_size=4)
        
        return a1