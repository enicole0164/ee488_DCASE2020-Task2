import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import Parameter


class ASDLoss(nn.Module):
    def __init__(self, reduction=True):
        super(ASDLoss, self).__init__()
        if reduction == True:
            self.ce = nn.CrossEntropyLoss()
        
        else:
            self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels):
        loss = self.ce(logits, labels)
        return loss

# ArcFace is referred to https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=200, s=30.0, m=0.7, sub=1, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.sub = sub
        self.weight = Parameter(torch.Tensor(out_features * sub, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        
        if self.sub > 1:
            cosine = cosine.view(-1, self.out_features, self.sub)
            cosine, _ = torch.max(cosine, dim=2)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output
    


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].[256,2,128]
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        # # Count class occurrences
        # unique_labels, counts = torch.unique(labels, return_counts=True)
        # class_counts = dict(zip(unique_labels.cpu().numpy(), counts.cpu().numpy()))
        # print("[Info] Class distribution in batch:", class_counts)

        mask = torch.eq(labels, labels.T).float().to(features.device)

        # TODO: normalize features in better manner
        features = F.normalize(features, dim=-1)  # Normalize along feature dim
        # Check for zero positive pairs
        positive_counts_per_sample = mask.sum(1) - 1  # Subtract self-pair
        num_samples_with_no_positives = (positive_counts_per_sample <= 0).sum()

        if num_samples_with_no_positives > 0:
            print(f"[Warning] {num_samples_with_no_positives.item()} samples have zero positive pairs.\n")

        # --------------------------------------------
        # Reshape features and determine anchor set
        # --------------------------------------------
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # shape: [batch_size * n_views, feature_dim]

        # Only use one view as anchor (first one)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1

        # Use all views as anchors (common in SupCon)
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # --------------------------------------
        # Numerical stability (avoid overflow)
        # --------------------------------------
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # --------------------------------------
        # Mask preparation
        # --------------------------------------
        # Expand the positive mask to match the shape of logits
        mask = mask.repeat(anchor_count, contrast_count)

         # Create logits mask to remove self-contrast (i.e., sample compared to itself)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(features.device),
            0
        )
        mask = mask * logits_mask

        # --------------------------------------
        # Log-softmax and loss computation
        # --------------------------------------
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Average log-probability over positive pairs
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)  #
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        
        # Final loss computation (temperature-scaled)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):

        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * 1e-4
        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                param.mul_(1 - self.wd)
