"""
modification made on the basis of link:https://github.com/liuyoude/STgram-MFN
"""
import math
from torch import nn
import torch
from losses import ArcMarginProduct

import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

import torch.distributed as dist

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
                 inp = 3,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(MobileFaceNet, self).__init__()

        self.conv1 = ConvBlock(inp, 64, 3, 2, 1) # When channel changes

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


class TgramNet(nn.Module): #STFT처럼 conv 기반으로 waveform을 mel-filter 적용한 결과로 반환
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
        # --- Gradient checkpointing ---
        x_t = self.tgramnet(x_wav).unsqueeze(1)
        
        x_mel_temp_att = self.temporal_attention(x_mel).unsqueeze(1)
       
        x = torch.cat((x_t, x_mel, x_mel_temp_att), dim=1)
        
        out, feature = self.mobilefacenet(x)

        feature = F.normalize(feature, dim=1)  # Normalize the feature vector
        
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

class SpecNet(nn.Module):
    """
    SpecNet: A concise CNN for processing frequency-domain information.
    Based on the paper's description: 3 layers of Conv1D + ReLU,
    processing Global Fourier Transform (FFT) of the input signal.
    """
    def __init__(self, 
                 signal_length=160000,  # 10 seconds at 16kHz (DCASE 2020 Task 2)
                 channels=128,          # Match other features' dimensionality
                 kernel_sizes=[9, 7, 5],  # Decreasing kernel sizes
                 strides=[8, 4, 8],       # Smaller strides than ASDNet
                 use_log_scale=True,
                 eps=1e-7):
        super(SpecNet, self).__init__()
        
        self.signal_length = signal_length
        self.use_log_scale = use_log_scale
        self.eps = eps
        
        # Following equation (7) and Figure 3: 3 layers of Conv1D + ReLU
        self.layers = nn.Sequential(
            # Layer 1: Conv1D + ReLU
            nn.Conv1d(1, 32, kernel_sizes[0], strides[0], 
                     padding=4, bias=True),
            nn.ReLU(inplace=True),
            
            # Layer 2: Conv1D + ReLU
            nn.Conv1d(32, 64, kernel_sizes[1], strides[1], 
                     padding=3, bias=True),
            nn.ReLU(inplace=True),
            
            # Layer 3: Conv1D + ReLU
            nn.Conv1d(64, 128, kernel_sizes[2], strides[2], 
                     padding=2, bias=True),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x_wav):
        """
        Args:
            x_wav: Input waveform tensor of shape (batch_size, 1, time_samples)
        
        Returns:
            Spectral features of shape (batch_size, channels, freq_features)
        """
        batch_size = x_wav.shape[0]
        
        # Extract waveform and ensure correct length
        x_flat = x_wav.squeeze(1)  # (batch_size, time_samples)
        
        # Pad or truncate to expected signal length
        if x_flat.shape[1] < self.signal_length:
            x_flat = F.pad(x_flat, (0, self.signal_length - x_flat.shape[1]))
        else:
            x_flat = x_flat[:, :self.signal_length]
        
        # Apply Global Fourier Transform (GFT) - equation (6) from paper
        # X[k] = Σ x(n)exp(-j2πkn/L) for k = 0, 1, ..., L-1
        fft_result = torch.fft.rfft(x_flat, n=self.signal_length)
        magnitude = torch.abs(fft_result)  # (batch_size, freq_bins)
        
        # Optional log scaling for better dynamic range
        if self.use_log_scale:
            magnitude = torch.log(magnitude + self.eps)
        
        # Reshape for Conv1d: (batch_size, 1, freq_bins)
        x = magnitude.unsqueeze(1)
        
        # Apply 3-layer CNN following equation (7)
        # X_j^l = δ(Σ X_i^(l-1) * W_ij^l + b_j^l)
        x = self.layers(x)
        
        return x  # (batch_size, channels, freq_features)

class TAST_SpecNetMFN(nn.Module):
    """
    Combines TASTgram (Temporal Attention + Tgram) with SpecNet features.
    This model uses:
    - Tgram for temporal features
    - SpecNet for spectral features
    - Temporal attention on mel spectrogram
    - Log-mel spectrogram
    All features are concatenated and processed through MobileFaceNet.
    """
    def __init__(self, 
                 num_classes,
                 mode,
                 c_dim=128,
                 win_len=1024,
                 hop_len=512,
                 signal_length=160000,
                 spec_kernel_sizes=[9, 7, 5],
                 spec_strides=[8, 4, 8],
                 bottleneck_setting=Mobilefacenet_bottleneck_setting,
                 use_arcface=True, 
                 m=0.7, 
                 s=30, 
                 sub=1):
        super().__init__()
        
        self.arcface = ArcMarginProduct(
            in_features=c_dim, 
            out_features=num_classes,
            m=m, s=s, sub=sub
        ) if use_arcface else use_arcface
        
        # TgramNet for temporal features (from TASTgram)
        self.tgramnet = TgramNet(mel_bins=c_dim, win_len=win_len, hop_len=hop_len)
        
        # SpecNet for spectral features
        self.specnet = SpecNet(
            signal_length=signal_length,
            channels=c_dim,
            kernel_sizes=spec_kernel_sizes,
            strides=spec_strides
        )
        
        # MobileFaceNet backbone - now processes 4 feature channels instead of 3
        self.mobilefacenet = MobileFaceNet(
            num_class=num_classes,
            bottleneck_setting=bottleneck_setting
        )
        
        self.mode = mode
        
        if mode not in ['arcface', 'arcmix', 'noisy_arcmix']:
            raise ValueError('Choose one of [arcface, arcmix, noisy_arcmix]')
        
        # Temporal attention for mel spectrogram (from TASTgram)
        self.temporal_attention = Temporal_Attention(feature_dim=c_dim)
        
        # # Adaptive pooling to match mel spectrogram time dimension (313)
        # self.adaptive_pool = nn.AdaptiveAvgPool1d(313)
        
        # Update conv1 in MobileFaceNet to accept 4 channels instead of 3
        self.mobilefacenet.conv1 = ConvBlock(4, 64, 3, 2, 1)
        
    def forward(self, x_wav, x_mel, label, train=True):
        # 1. Extract temporal features using TgramNet (from TASTgram)
        x_t = self.tgramnet(x_wav).unsqueeze(1)  # (batch, 1, 128, 313)
        
        # 2. Extract spectral features using SpecNet
        x_spec = self.specnet(x_wav).unsqueeze(1)  # (batch, 1, 128, 313)
        
        # 3. Apply temporal attention to mel spectrogram (from TASTgram)
        x_mel_att = self.temporal_attention(x_mel).unsqueeze(1)  # (batch, 1, 128, 313)
        
        # 4. Concatenate all features: [tgram, mel, mel_attention, specnet]
        # This gives us 4 feature channels combining TASTgram and SpecNet
        x = torch.cat((x_t, x_mel, x_mel_att, x_spec), dim=1)  # (batch, 4, 128, 313)
        
        # Pass through MobileFaceNet
        out, feature = self.mobilefacenet(x)
        feature = F.normalize(feature, dim=1)
        
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
        
    def getcosine(self, x_wav, x_mel):
        """
        Extract cosine similarity between Tgram and mel spectrogram features.
        This is useful for analyzing the relationship between temporal and spectral features.
        """
        # 1. Extract temporal features using TgramNet (from TASTgram)
        x_t = self.tgramnet(x_wav).unsqueeze(1)  # (batch, 1, 128, 313)
        
        # 2. Extract spectral features using SpecNet
        x_spec = self.specnet(x_wav).unsqueeze(1)  # (batch, 1, 128, 313)
        
        # 3. Apply temporal attention to mel spectrogram (from TASTgram)
        x_mel_att = self.temporal_attention(x_mel).unsqueeze(1)  # (batch, 1, 128, 313)
        
        # 4. Concatenate all features: [tgram, mel, mel_attention, specnet]
        # This gives us 4 feature channels combining TASTgram and SpecNet
        x = torch.cat((x_t, x_mel, x_mel_att, x_spec), dim=1)  # (batch, 4, 128, 313)
        
        # Pass through MobileFaceNet
        out, feature = self.mobilefacenet(x)
        
        cosine = F.linear(F.normalize(feature), F.normalize(self.arcface.weight))
        return cosine

    def all_gather(self, tensor):
        """Gather tensors from all processes. Use this before computing contrastive loss."""
        if not dist.is_initialized():
            return [tensor]
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return gathered

class TAST_SpecNetMFN_nrm(nn.Module):
    """
    Combines TASTgram (Temporal Attention + Tgram) with SpecNet features.
    This model uses:
    - Tgram for temporal features
    - SpecNet for spectral features
    - Temporal attention on mel spectrogram
    - Log-mel spectrogram
    All features are concatenated and processed through MobileFaceNet.
    """
    def __init__(self, 
                 num_classes,
                 mode,
                 c_dim=128,
                 win_len=1024,
                 hop_len=512,
                 signal_length=160000,
                 spec_kernel_sizes=[9, 7, 5],
                 spec_strides=[8, 4, 8],
                 bottleneck_setting=Mobilefacenet_bottleneck_setting,
                 use_arcface=True, 
                 m=0.7, 
                 s=30, 
                 sub=1):
        super().__init__()
        
        self.arcface = ArcMarginProduct(
            in_features=c_dim, 
            out_features=num_classes,
            m=m, s=s, sub=sub
        ) if use_arcface else use_arcface
        
        # TgramNet for temporal features (from TASTgram)
        self.tgramnet = TgramNet(mel_bins=c_dim, win_len=win_len, hop_len=hop_len)
        
        # SpecNet for spectral features
        self.specnet = SpecNet(
            signal_length=signal_length,
            channels=c_dim,
            kernel_sizes=spec_kernel_sizes,
            strides=spec_strides
        )
        
        # MobileFaceNet backbone - now processes 4 feature channels instead of 3
        self.mobilefacenet = MobileFaceNet(
            num_class=num_classes,
            bottleneck_setting=bottleneck_setting
        )
        
        self.mode = mode
        
        if mode not in ['arcface', 'arcmix', 'noisy_arcmix']:
            raise ValueError('Choose one of [arcface, arcmix, noisy_arcmix]')
        
        # Temporal attention for mel spectrogram (from TASTgram)
        self.temporal_attention = Temporal_Attention(feature_dim=c_dim)
        
        # # Adaptive pooling to match mel spectrogram time dimension (313)
        # self.adaptive_pool = nn.AdaptiveAvgPool1d(313)
        
        # Update conv1 in MobileFaceNet to accept 4 channels instead of 3
        self.mobilefacenet.conv1 = ConvBlock(4, 64, 3, 2, 1)
    
    def get_tgram(self, x_wav):
        return self.tgramnet(x_wav)
    
    def get_specnet(self, x_wav):
        return self.specnet(x_wav)
    
    def forward(self, x_wav, x_mel, label, train=True):
        # 1. Extract temporal features using TgramNet (from TASTgram)
        x_t = self.tgramnet(x_wav).unsqueeze(1)  # (batch, 1, 128, 313)
        x_t = F.normalize(x_t, dim=2)  # Normalize temporal features
        
        # 2. Extract spectral features using SpecNet
        x_spec = self.specnet(x_wav).unsqueeze(1)  # (batch, 1, 128, 313)
        x_spec = F.normalize(x_spec, dim=2)  # Normalize spectral features
        
        # 3. Apply temporal attention to mel spectrogram (from TASTgram)
        x_mel_att = self.temporal_attention(x_mel).unsqueeze(1)  # (batch, 1, 128, 313)
        x_mel_att = F.normalize(x_mel_att, dim=2) # Normalize mel attention features

        # 4. Normalize mel spectrogram
        x_mel = F.normalize(x_mel, dim=2)  # Normalize mel spectrogram
        
        # 5. Concatenate all features: [tgram, mel, mel_attention, specnet]
        # This gives us 4 feature channels combining TASTgram and SpecNet
        x = torch.cat((x_t, x_mel, x_mel_att, x_spec), dim=1)  # (batch, 4, 128, 313)
        
        # Pass through MobileFaceNet
        out, feature = self.mobilefacenet(x)
        feature = F.normalize(feature, dim=1)
        
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
        
    def getcosine(self, x_wav, x_mel):
        """
        Extract cosine similarity between Tgram and mel spectrogram features.
        This is useful for analyzing the relationship between temporal and spectral features.
        """
        # 1. Extract temporal features using TgramNet (from TASTgram)
        x_t = self.tgramnet(x_wav).unsqueeze(1)  # (batch, 1, 128, 313)
        
        # 2. Extract spectral features using SpecNet
        x_spec = self.specnet(x_wav).unsqueeze(1)  # (batch, 1, 128, 313)
        
        # 3. Apply temporal attention to mel spectrogram (from TASTgram)
        x_mel_att = self.temporal_attention(x_mel).unsqueeze(1)  # (batch, 1, 128, 313)
        
        # 4. Concatenate all features: [tgram, mel, mel_attention, specnet]
        # This gives us 4 feature channels combining TASTgram and SpecNet
        x = torch.cat((x_t, x_mel, x_mel_att, x_spec), dim=1)  # (batch, 4, 128, 313)
        
        # Pass through MobileFaceNet
        out, feature = self.mobilefacenet(x)
        
        cosine = F.linear(F.normalize(feature), F.normalize(self.arcface.weight))
        return cosine

    def all_gather(self, tensor):
        """Gather tensors from all processes. Use this before computing contrastive loss."""
        if not dist.is_initialized():
            return [tensor]
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return gathered

class TAST_SpecNetMFN_nrm2(nn.Module):
    """
    Combines TASTgram (Temporal Attention + Tgram) with SpecNet features.
    This model uses:
    - Tgram for temporal features
    - SpecNet for spectral features
    - Temporal attention on mel spectrogram
    - Log-mel spectrogram
    All features are concatenated and processed through MobileFaceNet.
    """
    def __init__(self, 
                 num_classes,
                 mode,
                 c_dim=128,
                 win_len=1024,
                 hop_len=512,
                 signal_length=160000,
                 spec_kernel_sizes=[9, 7, 5],
                 spec_strides=[8, 4, 8],
                 bottleneck_setting=Mobilefacenet_bottleneck_setting,
                 use_arcface=True, 
                 m=0.7, 
                 s=30, 
                 sub=1):
        super().__init__()
        
        self.arcface = ArcMarginProduct(
            in_features=c_dim, 
            out_features=num_classes,
            m=m, s=s, sub=sub
        ) if use_arcface else use_arcface
        
        # TgramNet for temporal features (from TASTgram)
        self.tgramnet = TgramNet(mel_bins=c_dim, win_len=win_len, hop_len=hop_len)
        
        # SpecNet for spectral features
        self.specnet = SpecNet(
            signal_length=signal_length,
            channels=c_dim,
            kernel_sizes=spec_kernel_sizes,
            strides=spec_strides
        )
        
        # MobileFaceNet backbone - now processes 4 feature channels instead of 3
        self.mobilefacenet = MobileFaceNet(
            num_class=num_classes,
            bottleneck_setting=bottleneck_setting
        )
        
        self.mode = mode
        
        if mode not in ['arcface', 'arcmix', 'noisy_arcmix']:
            raise ValueError('Choose one of [arcface, arcmix, noisy_arcmix]')
        
        # Temporal attention for mel spectrogram (from TASTgram)
        self.temporal_attention = Temporal_Attention(feature_dim=c_dim)
        
        # # Adaptive pooling to match mel spectrogram time dimension (313)
        # self.adaptive_pool = nn.AdaptiveAvgPool1d(313)
        
        # Update conv1 in MobileFaceNet to accept 4 channels instead of 3
        self.mobilefacenet.conv1 = ConvBlock(4, 64, 3, 2, 1)
            
    def get_tgram(self, x_wav):
        return self.tgramnet(x_wav)
    
    def get_specnet(self, x_wav):
        return self.specnet(x_wav)
    
    def forward(self, x_wav, x_mel, label, train=True):
        # 1. Extract temporal features using TgramNet (from TASTgram)
        x_t = self.tgramnet(x_wav).unsqueeze(1)  # (batch, 1, 128, 313)
        x_t = F.normalize(x_t, dim=(2,3))  # Normalize temporal features
        
        # 2. Extract spectral features using SpecNet
        x_spec = self.specnet(x_wav).unsqueeze(1)  # (batch, 1, 128, 313)
        x_spec = F.normalize(x_spec, dim=(2,3))  # Normalize spectral features
        
        # 3. Apply temporal attention to mel spectrogram (from TASTgram)
        x_mel_att = self.temporal_attention(x_mel).unsqueeze(1)  # (batch, 1, 128, 313)
        x_mel_att = F.normalize(x_mel_att, dim=(2,3)) # Normalize mel attention features

        # 4. Normalize mel spectrogram
        x_mel = F.normalize(x_mel, dim=(2,3))  # Normalize mel spectrogram
        
        # 5. Concatenate all features: [tgram, mel, mel_attention, specnet]
        # This gives us 4 feature channels combining TASTgram and SpecNet
        x = torch.cat((x_t, x_mel, x_mel_att, x_spec), dim=1)  # (batch, 4, 128, 313)
        
        # Pass through MobileFaceNet
        out, feature = self.mobilefacenet(x)
        feature = F.normalize(feature, dim=1)
        
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
        
    def getcosine(self, x_wav, x_mel):
        """
        Extract cosine similarity between Tgram and mel spectrogram features.
        This is useful for analyzing the relationship between temporal and spectral features.
        """
        # 1. Extract temporal features using TgramNet (from TASTgram)
        x_t = self.tgramnet(x_wav).unsqueeze(1)  # (batch, 1, 128, 313)
        
        # 2. Extract spectral features using SpecNet
        x_spec = self.specnet(x_wav).unsqueeze(1)  # (batch, 1, 128, 313)
        
        # 3. Apply temporal attention to mel spectrogram (from TASTgram)
        x_mel_att = self.temporal_attention(x_mel).unsqueeze(1)  # (batch, 1, 128, 313)
        
        # 4. Concatenate all features: [tgram, mel, mel_attention, specnet]
        # This gives us 4 feature channels combining TASTgram and SpecNet
        x = torch.cat((x_t, x_mel, x_mel_att, x_spec), dim=1)  # (batch, 4, 128, 313)
        
        # Pass through MobileFaceNet
        out, feature = self.mobilefacenet(x)
        
        cosine = F.linear(F.normalize(feature), F.normalize(self.arcface.weight))
        return cosine

    def all_gather(self, tensor):
        """Gather tensors from all processes. Use this before computing contrastive loss."""
        if not dist.is_initialized():
            return [tensor]
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return gathered
