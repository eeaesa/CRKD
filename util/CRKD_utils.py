import numpy as np
import torch
from skimage.measure import regionprops
from torch.nn import functional as F
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat
import torch.distributed as dist
from copy import deepcopy
import random

class ProjectionHead_2d(nn.Module):
    """PyTorch version of projection_head"""

    def __init__(self,
                 num_input_channels,
                 num_projection_channels=256,
                 num_projection_layers=3,
                 ):
        super().__init__()
        self.layers = nn.ModuleList()

        # Intermediate layers
        for _ in range(num_projection_layers - 1):
            self.layers.append(nn.Conv2d(num_input_channels, num_projection_channels, kernel_size=1))
            self.layers.append(nn.BatchNorm2d(num_projection_channels))
            self.layers.append(nn.ReLU(inplace=True))
            num_input_channels = num_projection_channels

        # Final layer
        self.final_conv = nn.Conv2d(num_input_channels, num_projection_channels, kernel_size=1)

    def forward(self, x):
        '''
        input:(b, d, h, w)
        output:(b, d, h, w)
        '''
        for layer in self.layers:
            x = layer(x)
        x = self.final_conv(x)
        return F.normalize(x, p=2, dim=1)

class ProjectionHead_3d(nn.Module):

    def __init__(self,
                 num_input_channels,
                 num_projection_channels=256,
                 num_projection_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()

        # Intermediate layers
        for _ in range(num_projection_layers - 1):
            self.layers.append(nn.Conv3d(num_input_channels, num_projection_channels, kernel_size=1))
            self.layers.append(nn.BatchNorm3d(num_projection_channels))
            self.layers.append(nn.ReLU(inplace=True))
            num_input_channels = num_projection_channels

        # Final layer
        self.final_conv = nn.Conv3d(num_input_channels, num_projection_channels, kernel_size=1)

    def forward(self, x):
        '''
        input: (b, c_in, h, w, d)
        output: (b, c_proj, h, w, d)
        '''
        if x.dim() == 5:
            if x.shape[2] > x.shape[3]:
                x = x.permute(0, 1, 3, 4, 2)  # (B, C, H, W, D)

        for layer in self.layers:
            x = layer(x)
        x = self.final_conv(x)
        return F.normalize(x, p=2, dim=1)

def entry_consistency_loss(pred_u_s, pred_u_fp,
                                beta=1.0, distance_type='kl'):
    '''
    pred_u_s: [B, C, H, W, D]
    pred_u_fp: [B, C, H, W, D]
    beta: temp
    distance_type: 'l1' or 'kl'
    '''
    pred_u_s = pred_u_s.softmax(dim=1)
    pred_u_fp = pred_u_fp.softmax(dim=1)
    preds = [pred_u_s, pred_u_fp]
    M = len(preds)

    entropies, max_probs = [], []
    for prob in preds:
        entropy = - (prob * torch.log(prob + 1e-8)).sum(dim=1)
        max_prob = torch.max(prob, dim=1)[0]
        entropies.append(entropy)
        max_probs.append(max_prob)

    entropies = torch.stack(entropies, dim=0)   # [M, B, H, W, D]
    max_probs = torch.stack(max_probs, dim=0)   # [M, B, H, W, D]

    entropy_weights = torch.exp(-entropies)
    combined_weights = entropy_weights * (max_probs ** beta)
    alpha = combined_weights / (combined_weights.sum(dim=0, keepdim=True) + 1e-8)

    prob_stack = torch.stack(preds, dim=0)  # [M, B, C, H, W, D]
    alpha_exp = alpha.unsqueeze(2)          # [M, B, 1, H, W, D]
    aggregated_pred = (prob_stack * alpha_exp).sum(dim=0)  # [B, C, H, W, D]

    if distance_type == 'l1':
        diff = torch.abs(prob_stack - aggregated_pred.unsqueeze(0))  # [M, B, C, H, W, D]
        distance = diff.sum(dim=2)  # [M, B, H, W, D]
    elif distance_type == 'kl':
        ratio = (prob_stack + 1e-8) / (aggregated_pred.unsqueeze(0) + 1e-8)
        kl = prob_stack * torch.log(ratio)
        distance = kl.sum(dim=2)  # [M, B, H, W, D]
    else:
        raise ValueError(f"Unsupported distance_type: {distance_type}")

    final_weight = combined_weights / (combined_weights.sum(dim=0, keepdim=True) + 1e-8)
    numerator = (final_weight * distance).sum(dim=0)  # [B, H, W, D]
    denominator = final_weight.sum(dim=0)  # [b, h, w, d]
    numerator = numerator / (denominator + 1e-8)  # [b, h, w, d]
    loss = numerator.mean().to(pred_u_s.device)

    return loss

class CSD_2d(nn.Module):
    def __init__(self,
                 num_class=2,
                 K=256,
                 threshold=0.75,
                 temp=1.0,
                 overlap=False,
                 use_threshold=False,
                 use_TP=False,
                 ):
        super().__init__()
        self.num_class = num_class
        self.K = K
        self.threshold = threshold
        self.temp = temp
        self.overLapSample = overlap
        self.use_threshold = use_threshold
        self.use_TP = use_TP

    def forward(self,
                feat_x, pred_gt, gt,
                feat_u_s,
                feat_u_fp,
                logits_u, label_u):

        '''
        feat_x: [B, D, H, W]
        pred_gt: [B, H, W]
        gt: [B, H, W]
        feat_u_s: [B, D, H, W]
        feat_u_fp: [B, D, H, W]
        logits_u: [B, H, W]
        label_u: [B, H, W]
        '''

        # mask_valid = (logits_u >= logits_u.mean()).float()

        B, D, H, W = feat_x.shape

        gt = F.interpolate(
            gt.unsqueeze(1).float(),
            size=(H, W),
            mode='nearest'
        ).squeeze(1).long()
        pred_gt = F.interpolate(
            pred_gt.unsqueeze(1).float(),
            size=(H, W),
            mode='nearest'
        ).squeeze(1).long()
        logits_u = F.interpolate(logits_u.unsqueeze(1), size=(H, W),
                                   mode='bilinear', align_corners=True).squeeze(1)
        label_u = F.interpolate(
            label_u.float().unsqueeze(1),
            size=(H, W),
            mode='nearest'
        ).squeeze(1).long()

        feat_l, label_l = self.sample_label_feats(feat_x, pred_gt, gt)
        mask_l = ~torch.isnan(feat_l).all(dim=1)
        feat_l = feat_l[mask_l]
        label_l = label_l[mask_l]

        features_list, labels_list = [], []

        base_samples =  self.K // self.num_class
        remainder = self.K % self.num_class

        f_s = feat_u_s.permute(0, 2, 3, 1) # [B, H, W, D]
        f_fp = feat_u_fp.permute(0, 2, 3, 1) # [B, H, W, D]

        for cls_idx in range(self.num_class):
            if self.use_threshold:
                mask = (label_u == cls_idx) & (logits_u >= self.threshold)
            else:
                mask = (label_u == cls_idx) & (logits_u >= logits_u.mean())
            mask_f = mask
            if self.overLapSample:
                mask_s, mask_fp = mask_f, mask_f
            else:
                mask_s, mask_fp = self.split_mask(mask_f)

            s_features = f_s[mask_s]
            fp_features = f_fp[mask_fp]

            if cls_idx == self.num_class - 1:
                n_samples = base_samples + remainder
            else:
                n_samples = base_samples

            feat_s, label_s = self.sample_unlabel_feats(
                s_features, n_samples, cls_idx
            )

            feat_fp, label_fp = self.sample_unlabel_feats(
                fp_features, n_samples, cls_idx
            )

            features_u = torch.cat([feat_s, feat_fp])
            labels_u = torch.cat([label_s, label_fp])
            features_list.append(features_u)
            labels_list.append(labels_u)

        sampled_features = torch.cat(features_list, dim=0)  # [2(N+num_classes), D_feat]
        sampled_labels = torch.cat(labels_list, dim=0)  # [2(N+num_classes),]

        mask_u = ~torch.isnan(sampled_features).all(dim=1)
        feats_u = sampled_features[mask_u]
        labels_u = sampled_labels[mask_u]

        if feat_l.size(0) == 0 or feats_u.size(0) == 0:
            loss = torch.tensor(0.0)
        else:
            mask = torch.eq(label_l.unsqueeze(1), labels_u.unsqueeze(1).T).float().to(label_l.device)
            anchor_dot_contrast = torch.div(torch.matmul(feat_l, feats_u.T), self.temp)
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
            mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
            loss = -mean_log_prob_pos.mean()

        return loss.to(feat_x.device)

    def sample_label_feats(self, feat_x, pred_gt, gt):
        '''
        feat_x: [B, D, H, W]
        pred_gt: [B, H, W]
        gt: [B, H, W]

        out:
            features: [N+C, D_feat]
            labels: [N+C]
        '''
        device = feat_x.device
        B, D, H, W = feat_x.shape
        C = self.num_class
        base_samples = self.K // C
        remainder = self.K % C

        f_flat = feat_x.permute(0, 2, 3, 1).reshape(-1, D)  # [B*H*W, D]
        pred_flat = pred_gt.reshape(-1)  # [B*H*W, D]
        gt_flat = gt.reshape(-1)  # [B*H*W, D]

        features_list, labels_list = [], []
        for cls in range(C):
            if cls == C - 1:
                n_samples = base_samples + remainder
            else:
                n_samples = base_samples

            if self.use_TP:
                mask = (pred_flat == cls) & (gt_flat == cls)
            else:
                mask = (pred_flat == cls)

            class_features = f_flat[mask]  # [num_valid, D_feat]
            num_valid = class_features.size(0)

            if num_valid == 0:
                zero_feature = torch.full((n_samples + 1, D), torch.nan, device=device)
                zero_label = torch.full((n_samples + 1,), cls, device=device)
                features_list.append(zero_feature)
                labels_list.append(zero_label)
                continue

            if num_valid >= n_samples:
                indices = torch.randint(0, num_valid, (n_samples,), device=device)
            else:
                repeat_times = (n_samples // num_valid) + 1
                indices = torch.arange(num_valid, device=device).repeat(repeat_times)[:n_samples]

            sampled = class_features[indices]  # [n_samples, D_feat]

            mean_feature = class_features.mean(dim=0, keepdim=True)  # [1, D_feat]
            combined = torch.cat([sampled, mean_feature], dim=0)  # [n_samples+1, D_feat]

            features_list.append(combined)
            labels_list.append(torch.full((n_samples + 1,), cls, device=device))

        features = torch.cat(features_list, dim=0)  # [N+C, D_feat]
        labels = torch.cat(labels_list, dim=0)  # [N+C]

        return features, labels

    def sample_unlabel_feats(self, feat, n_samples, cls_idx):
        '''
        feat=[num, D]
        '''
        device = feat.device
        num_valid, D_feat = feat.size()
        num_valid = feat.size(0)

        if num_valid == 0:
            zero_feature = torch.full((n_samples + 1, D_feat), torch.nan, device=device)
            zero_label = torch.full((n_samples + 1,), cls_idx, device=device)
            return zero_feature, zero_label

        if num_valid >= n_samples:
            indices = torch.randint(0, num_valid, (n_samples,), device=device)
        else:
            repeat_times = (n_samples // num_valid) + 1
            indices = torch.arange(num_valid, device=device).repeat(repeat_times)[:n_samples]

        sampled = feat[indices]  # [n_samples, D_feat]

        mean_feature = feat.mean(dim=0, keepdim=True)  # [1, D_feat]
        combined_feature = torch.cat([sampled, mean_feature], dim=0)  # [n_samples+1, D_feat]
        labels_feature = torch.full((n_samples + 1,), cls_idx, device=device)

        return combined_feature, labels_feature

    def split_mask(self, mask_entry):
        """
        Args:
            mask_entry: Boolean tensor of shape [B, h, w]
        Returns:
            mask_1, mask_2: Two boolean tensors, each containing half of the True values.
        """
        true_indices = torch.nonzero(mask_entry)
        N = true_indices.shape[0]
        half = N // 2

        perm = torch.randperm(N)
        true_indices = true_indices[perm]

        indices_1 = true_indices[:half]
        indices_2 = true_indices[half:]

        mask_1 = torch.zeros_like(mask_entry)
        mask_2 = torch.zeros_like(mask_entry)

        mask_1[tuple(indices_1.T)] = True
        mask_2[tuple(indices_2.T)] = True

        return mask_1, mask_2

class CSD_3d(nn.Module):
    def __init__(self,
                 num_class=2,
                 K=256,
                 threshold=0.75,
                 temp=1.0,
                 overlap=True,
                 use_threshold=False,
                 use_TP=False,
                 ):
        super().__init__()
        self.num_class = num_class
        self.K = K
        self.threshold = threshold
        self.temp = temp
        self.overLapSample = overlap
        self.use_threshold = use_threshold
        self.use_TP = use_TP

    def forward(self,
                feat_x, pred_gt, gt,
                feat_u_s,
                feat_u_fp,
                logits_u, label_u):

        '''
        feat_x: [B, F, H, W, D]
        pred_gt: [B, H, W, D]
        gt: [B, H, W, D]
        pred_u_s: [B, C, H, W, D]
        feat_u_s: [B, F, H, W, D]
        pred_u_fp: [B, C, H, W, D]
        feat_u_fp: [B, F, H, W, D]
        logits_u: [B, H, W, D]
        label_u: [B, H, W, D]
        '''

        B, D, H, W, Depth = feat_x.shape

        gt = F.interpolate(
            gt.unsqueeze(1).float(),
            size=(H, W, Depth),
            mode='nearest'
        ).squeeze(1).long()
        pred_gt = F.interpolate(
            pred_gt.unsqueeze(1).float(),
            size=(H, W, Depth),
            mode='nearest'
        ).squeeze(1).long()
        logits_u = F.interpolate(logits_u.unsqueeze(1), size=(H, W, Depth),
                                   mode='trilinear', align_corners=False).squeeze(1)
        label_u = F.interpolate(
            label_u.unsqueeze(1).float(),
            size=(H, W, Depth),
            mode='nearest'
        ).squeeze(1).long()

        feat_l, label_l = self.sample_label_feats(feat_x, pred_gt, gt)
        mask_l = ~torch.isnan(feat_l).all(dim=1)
        feat_l = feat_l[mask_l]
        label_l = label_l[mask_l]

        features_list, labels_list = [], []

        base_samples =  self.K // self.num_class
        remainder = self.K % self.num_class

        f_s = feat_u_s.permute(0, 2, 3, 4, 1) # [B, H, W, Depth, D]
        f_fp = feat_u_fp.permute(0, 2, 3, 4, 1) # [B, H, W, Depth, D]

        for cls_idx in range(self.num_class):
            if self.use_threshold:
                mask = (label_u == cls_idx) & (logits_u >= self.threshold)
            else:
                mask = (label_u == cls_idx) & (logits_u >= logits_u.mean())
            mask_f = mask
            if self.overLapSample:
                mask_s, mask_fp = self.split_mask(mask_f)
            else:
                mask_s, mask_fp = mask_f, mask_f

            s_features = f_s[mask_s]  # [num_1, D_feat]
            fp_features = f_fp[mask_fp]  # [num_2, D_feat]

            if cls_idx == self.num_class - 1:
                n_samples = base_samples + remainder
            else:
                n_samples = base_samples

            feat_s, label_s = self.sample_unlabel_feats(
                s_features, n_samples, cls_idx
            )

            feat_fp, label_fp = self.sample_unlabel_feats(
                fp_features, n_samples, cls_idx
            )

            features_u = torch.cat([feat_s, feat_fp])
            labels_u = torch.cat([label_s, label_fp])
            features_list.append(features_u)
            labels_list.append(labels_u)

        sampled_features = torch.cat(features_list, dim=0)  # [2(N+num_classes), D_feat]
        sampled_labels = torch.cat(labels_list, dim=0)  # [2(N+num_classes),]

        mask_u = ~torch.isnan(sampled_features).all(dim=1)
        feats_u = sampled_features[mask_u]
        labels_u = sampled_labels[mask_u]

        if feat_l.size(0) == 0 or feats_u.size(0) == 0:
            loss = torch.tensor(0.0)
        else:
            mask = torch.eq(label_l.unsqueeze(1), labels_u.unsqueeze(1).T).float().to(label_l.device)
            anchor_dot_contrast = torch.div(torch.matmul(feat_l, feats_u.T), self.temp)
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
            mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
            loss = -mean_log_prob_pos.mean()

        return loss.to(feat_x.device)

    def sample_label_feats(self, feat_x, pred_gt, gt):
        '''

        feat_x: [B, D, H, W, Depth]
        pred_gt: [B, H, W, Depth]
        gt: [B, H, W, Depth]

    out:
        features: [N+C, D_feat]
        labels: [N+C]
        '''
        device = feat_x.device
        B, D_feat, H, W, D = feat_x.shape
        C = self.num_class
        base_samples = self.K // C
        remainder = self.K % C

        # 展平数据
        f_flat = feat_x.permute(0, 2, 3, 4, 1).reshape(-1, D, D_feat)  # [B*H*W, D, D_feat]
        pred_flat = pred_gt.reshape(-1, D)  # [B*H*W, D]
        gt_flat = gt.reshape(-1, D)  # [B*H*W, D]

        features_list, labels_list = [], []
        for cls in range(C):
            if cls == C - 1:
                n_samples = base_samples + remainder
            else:
                n_samples = base_samples

            if self.use_TP:
                mask = (pred_flat == cls) & (gt_flat == cls)
            else:
                mask = (pred_flat == cls)

            class_features = f_flat[mask]  # [num_valid, D_feat]
            num_valid = class_features.size(0)

            if num_valid == 0:
                zero_feature = torch.full((n_samples + 1, D_feat), torch.nan, device=device)
                zero_label = torch.full((n_samples + 1,), cls, device=device)
                features_list.append(zero_feature)
                labels_list.append(zero_label)
                continue

            if num_valid >= n_samples:
                indices = torch.randint(0, num_valid, (n_samples,), device=device)
            else:
                repeat_times = (n_samples // num_valid) + 1
                indices = torch.arange(num_valid, device=device).repeat(repeat_times)[:n_samples]

            sampled = class_features[indices]  # [n_samples, D_feat]

            mean_feature = class_features.mean(dim=0, keepdim=True)  # [1, D_feat]
            combined = torch.cat([sampled, mean_feature], dim=0)  # [n_samples+1, D_feat]

            features_list.append(combined)
            labels_list.append(torch.full((n_samples + 1,), cls, device=device))

        features = torch.cat(features_list, dim=0)  # [N+C, D_feat]
        labels = torch.cat(labels_list, dim=0)  # [N+C]

        return features, labels

    def sample_unlabel_feats(self, feat, n_samples, cls_idx):
        '''
        feat=[num, D]
        '''
        device = feat.device
        num_valid, D_feat = feat.size()
        num_valid = feat.size(0)

        if num_valid == 0:
            zero_feature = torch.full((n_samples + 1, D_feat), torch.nan, device=device)
            zero_label = torch.full((n_samples + 1,), cls_idx, device=device)
            return zero_feature, zero_label

        if num_valid >= n_samples:
            indices = torch.randint(0, num_valid, (n_samples,), device=device)
        else:
            repeat_times = (n_samples // num_valid) + 1
            indices = torch.arange(num_valid, device=device).repeat(repeat_times)[:n_samples]

        sampled = feat[indices]  # [n_samples, D_feat]

        mean_feature = feat.mean(dim=0, keepdim=True)  # [1, D_feat]
        combined_feature = torch.cat([sampled, mean_feature], dim=0)  # [n_samples+1, D_feat]
        labels_feature = torch.full((n_samples + 1,), cls_idx, device=device)

        return combined_feature, labels_feature

    def split_mask(self, mask_entry):
        """
        Args:
            mask_entry: Boolean tensor of shape [B, h, w, d]
        Returns:
            mask_1, mask_2: Two boolean tensors, each containing half of the True values.
        """
        true_indices = torch.nonzero(mask_entry)
        N = true_indices.shape[0]
        half = N // 2

        perm = torch.randperm(N)
        true_indices = true_indices[perm]

        indices_1 = true_indices[:half]
        indices_2 = true_indices[half:]

        mask_1 = torch.zeros_like(mask_entry)
        mask_2 = torch.zeros_like(mask_entry)

        mask_1[tuple(indices_1.T)] = True
        mask_2[tuple(indices_2.T)] = True

        return mask_1, mask_2


class CrossKDLoss_2d(nn.Module):

    def __init__(self,
                 use_pixel=True,
                 use_feature=True,
                 feature_mode='cosine',
                 T=1.0,
                 temperature=0.07
                 ):
        super().__init__()
        self.use_pixel = use_pixel
        self.use_feature = use_feature
        self.feature_mode = feature_mode
        self.T = T
        self.temperature = temperature

        if use_pixel:
            self.kd_loss = KDLoss(T)

    def forward(self,
                pred_u_s: torch.Tensor,  # [B,C,H,W]
                pred_u_fp: torch.Tensor,  # [B,C,H,W]
                feat_u_s: torch.Tensor,  # [B,D,H,W]
                feat_u_fp: torch.Tensor,
                compare_matrix: torch.Tensor,  # [B,H,W]
                conf_mask: torch.Tensor,  # [B,H,W]
                ) -> torch.Tensor:

        device = pred_u_s.device

        loss_pix_fp_to_s = torch.tensor(0.0).to(device)
        loss_pix_s_to_fp = torch.tensor(0.0).to(device)
        loss_feat_fp_to_s = torch.tensor(0.0).to(device)
        loss_feat_s_to_fp = torch.tensor(0.0).to(device)

        if conf_mask.mean() > 0:
            mask_s_better = compare_matrix * conf_mask  # [B, H, W]
            mask_fp_better = (1 - compare_matrix) * conf_mask  # [B, H, W]

            if self.use_pixel:
                loss_fp_to_s = self.kd_loss(pred_u_fp, pred_u_s.detach()).mean(dim=1)  # [B, H, W]
                loss_s_to_fp = self.kd_loss(pred_u_s, pred_u_fp.detach()).mean(dim=1)  # [B, H, W]

                loss_pix_fp_to_s = (loss_fp_to_s * mask_s_better).sum() / (mask_s_better.sum() + 1e-8)
                loss_pix_s_to_fp = (loss_s_to_fp * mask_fp_better).sum() / (mask_fp_better.sum() + 1e-8)

            if self.use_feature:
                B, D, H, W = feat_u_s.shape

                f_s_better = F.interpolate(mask_s_better.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)
                f_fp_better = F.interpolate(mask_fp_better.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)
                feat_s_norm = F.normalize(feat_u_s, p=2, dim=1)
                feat_fp_norm = F.normalize(feat_u_fp, p=2, dim=1)
                sim_matrix = torch.einsum("bdhw,bdhw->bhw", feat_s_norm, feat_fp_norm) / self.temperature  # [B, H, W]
                sim_matrix = 1 - sim_matrix

                loss_feat_fp_to_s = (sim_matrix * f_s_better).sum() / (f_s_better.sum() + 1e-8)
                loss_feat_s_to_fp = (sim_matrix * f_fp_better).sum() / (f_fp_better.sum() + 1e-8)

        loss = {
            'loss_pix_fp_to_s': loss_pix_fp_to_s,
            "loss_pix_s_to_fp": loss_pix_s_to_fp,
            'loss_feat_fp_to_s': loss_feat_fp_to_s,
            'loss_feat_s_to_fp': loss_feat_s_to_fp,
        }

        return loss


class CrossKDLoss_3d(nn.Module):
    def __init__(self,
                 use_pixel=True,
                 use_feature=True,
                 feature_mode='cosine',
                 T=1.0,
                 temperature=0.07):
        super().__init__()
        self.use_pixel = use_pixel
        self.use_feature = use_feature
        self.feature_mode = feature_mode
        self.T = T
        self.temperature = temperature

        if use_pixel:
            self.kd_loss = KDLoss(T)

    def forward(self,
                pred_u_s: torch.Tensor,  # [B, C, H, W, Depth]
                pred_u_fp: torch.Tensor,  # [B, C, H, W, Depth]
                feat_u_s: torch.Tensor,   # [B, D_feat, H, W, Depth]
                feat_u_fp: torch.Tensor,  # [B, D_feat, H, W, Depth]
                compare_matrix: torch.Tensor,  # [B, H, W, Depth]
                conf_mask: torch.Tensor,  # [B, H, W, Depth]
                ) -> torch.Tensor:

        device = pred_u_s.device

        loss_pix_fp_to_s = torch.tensor(0.0).to(device)
        loss_pix_s_to_fp = torch.tensor(0.0).to(device)
        loss_feat_fp_to_s = torch.tensor(0.0).to(device)
        loss_feat_s_to_fp = torch.tensor(0.0).to(device)

        if conf_mask.float().mean() > 0:
            mask_s_better = compare_matrix * conf_mask  # [B, H, W, Depth]
            mask_fp_better = (1 - compare_matrix) * conf_mask  # [B, H, W, Depth]

            if self.use_pixel:
                loss_fp_to_s = self.kd_loss(pred_u_fp, pred_u_s.detach()).mean(dim=1)  # [B, H, W, Depth]
                loss_s_to_fp = self.kd_loss(pred_u_s, pred_u_fp.detach()).mean(dim=1)  # [B, H, W, Depth]
                loss_pix_fp_to_s = (loss_fp_to_s * mask_s_better).sum() / (mask_s_better.sum() + 1e-8)
                loss_pix_s_to_fp = (loss_s_to_fp * mask_fp_better).sum() / (mask_fp_better.sum() + 1e-8)

            if self.use_feature:
                B, D_feat, H, W, Depth = feat_u_s.shape

                f_s_better = F.interpolate(mask_s_better.unsqueeze(1), size=(H, W, Depth), mode='nearest').squeeze(1)
                f_fp_better = F.interpolate(mask_fp_better.unsqueeze(1), size=(H, W, Depth), mode='nearest').squeeze(1)

                feat_s_norm = F.normalize(feat_u_s, p=2, dim=1)  # [B, D_feat, H, W, Depth]
                feat_fp_norm = F.normalize(feat_u_fp, p=2, dim=1)  # [B, D_feat, H, W, Depth]

                sim_matrix = torch.einsum("bchwd,bchwd->bhwd", feat_s_norm, feat_fp_norm) / self.temperature  # [B, H, W, Depth]
                sim_matrix = 1 - sim_matrix  # [B, H, W, Depth]

                loss_feat_fp_to_s = (sim_matrix * f_s_better).sum() / (f_s_better.sum() + 1e-8)
                loss_feat_s_to_fp = (sim_matrix * f_fp_better).sum() / (f_fp_better.sum() + 1e-8)

        loss = {
            'loss_pix_fp_to_s': loss_pix_fp_to_s,
            "loss_pix_s_to_fp": loss_pix_s_to_fp,
            'loss_feat_fp_to_s': loss_feat_fp_to_s,
            'loss_feat_s_to_fp': loss_feat_s_to_fp,
        }

        return loss

class KDLoss(nn.Module):
    def __init__(self,
                 T=1.0,
                 reduction="none"
                 ):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, out_s, out_t, dim=1):
        loss = (
            F.kl_div(
                F.log_softmax(out_s / self.T, dim=dim),
                F.softmax(out_t / self.T, dim=dim),
                reduction=self.reduction
            )
            * self.T
            * self.T
        )
        return loss

def batch_symmetric_random_cutmix_2d(
    x_u_w, x_u_s, pred_u_w,
    prob=0.5,
    size_min=0.02, size_max=0.4,
    ratio_1=0.3, ratio_2=1 / 0.3,
    softmax=False
):
    b, c, h, w = x_u_w.shape

    if softmax:
        pred_u_w = F.softmax(pred_u_w, dim=1)

    logits_u, label_u = torch.max(pred_u_w, dim=1)  # [B, H, W]

    new_x_u_w = x_u_w.clone()
    new_x_u_s = x_u_s.clone()
    new_label_u = label_u.clone()
    new_logits_u = logits_u.clone()

    indices = torch.randperm(b)

    for i in range(0, b - 1, 2):
        idx1 = indices[i]
        idx2 = indices[i + 1]

        mask = obtain_cutmix_box_2d(
            img_size=(h, w),
            p=prob,
            size_min=size_min,
            size_max=size_max,
            ratio_1=ratio_1,
            ratio_2=ratio_2
        )

        if mask.sum() == 0:
            continue

        mask = mask.to(x_u_w.device).unsqueeze(0)  # [1, H, W]

        # A ← B
        new_x_u_w[idx1] = x_u_w[idx1] * (1 - mask) + x_u_w[idx2] * mask
        new_x_u_s[idx1] = x_u_s[idx1] * (1 - mask) + x_u_s[idx2] * mask
        new_label_u[idx1] = label_u[idx1] * (1 - mask[0]) + label_u[idx2] * mask[0]
        new_logits_u[idx1] = logits_u[idx1] * (1 - mask[0]) + logits_u[idx2] * mask[0]

        # B ← A
        new_x_u_w[idx2] = x_u_w[idx2] * (1 - mask) + x_u_w[idx1] * mask
        new_x_u_s[idx2] = x_u_s[idx2] * (1 - mask) + x_u_s[idx1] * mask
        new_label_u[idx2] = label_u[idx2] * (1 - mask[0]) + label_u[idx1] * mask[0]
        new_logits_u[idx2] = logits_u[idx2] * (1 - mask[0]) + logits_u[idx1] * mask[0]

    return new_x_u_w, new_x_u_s, new_label_u, new_logits_u

def obtain_cutmix_box_2d(
    img_size=(112, 112),
    p=0.5,
    size_min=0.02,
    size_max=0.4,
    ratio_1=0.3,
    ratio_2=1 / 0.3,
):
    mask = torch.zeros(img_size)
    img_h, img_w = img_size

    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_h * img_w
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))

        if cutmix_w == 0 or cutmix_h == 0:
            return mask

        if cutmix_w > img_w or cutmix_h > img_h:
            continue

        x = np.random.randint(0, img_w - cutmix_w + 1)
        y = np.random.randint(0, img_h - cutmix_h + 1)

        break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1
    return mask

def obtain_cutmix_box_3d(
    img_size=(112, 112, 80),
    p=0.5,
    size_min=0.02,
    size_max=0.4,
    ratio_1=0.3,
    ratio_2=1 / 0.3,
):
    mask = torch.zeros(img_size)
    img_size_x, img_size_y, img_size_z = img_size

    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size_x * img_size_y * img_size_z
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.power(size / ratio, 1 / 3))
        cutmix_h = int(np.power(size * ratio, 1 / 3))
        cutmix_d = int(np.power(size, 1 / 3))

        if cutmix_w == 0 or cutmix_h == 0 or cutmix_d == 0:
            return mask

        x = np.random.randint(0, img_size_x - cutmix_w + 1)
        y = np.random.randint(0, img_size_y - cutmix_h + 1)
        z = np.random.randint(0, img_size_z - cutmix_d + 1)

        if (
            x + cutmix_w <= img_size_x
            and y + cutmix_h <= img_size_y
            and z + cutmix_d <= img_size_z
        ):
            break

    mask[z : z + cutmix_d, y : y + cutmix_h, x : x + cutmix_w] = 1
    return mask


def batch_symmetric_random_cutmix_3d(
    x_u_w, x_u_s, pred_u_w,
    prob=0.5,
    size_min=0.02, size_max=0.4,
    ratio_1=0.3, ratio_2=1 / 0.3,
    softmax=False
):
    b, c, h, w, d = x_u_w.shape

    if softmax:
        pred_u_w = F.softmax(pred_u_w, dim=1)

    logits_u, label_u = torch.max(pred_u_w, dim=1)  # [B, H, W, D]

    new_x_u_w = x_u_w.clone()
    new_x_u_s = x_u_s.clone()
    new_label_u = label_u.clone()
    new_logits_u = logits_u.clone()

    indices = torch.randperm(b)

    for i in range(0, b - 1, 2):
        idx1 = indices[i]
        idx2 = indices[i + 1]

        mask = obtain_cutmix_box_3d(img_size=(h, w, d), p=prob,
                                 size_min=size_min, size_max=size_max,
                                 ratio_1=ratio_1, ratio_2=ratio_2)  # [D, W, H]

        if mask.sum() == 0:
            continue

        mask = mask.to(x_u_w.device).unsqueeze(0)  # [1, H, W, D]

        # A ← B
        new_x_u_w[idx1] = x_u_w[idx1] * (1 - mask) + x_u_w[idx2] * mask
        new_x_u_s[idx1] = x_u_s[idx1] * (1 - mask) + x_u_s[idx2] * mask
        new_label_u[idx1] = label_u[idx1] * (1 - mask[0]) + label_u[idx2] * mask[0]
        new_logits_u[idx1] = logits_u[idx1] * (1 - mask[0]) + logits_u[idx2] * mask[0]

        # B ← A
        new_x_u_w[idx2] = x_u_w[idx2] * (1 - mask) + x_u_w[idx1] * mask
        new_x_u_s[idx2] = x_u_s[idx2] * (1 - mask) + x_u_s[idx1] * mask
        new_label_u[idx2] = label_u[idx2] * (1 - mask[0]) + label_u[idx1] * mask[0]
        new_logits_u[idx2] = logits_u[idx2] * (1 - mask[0]) + logits_u[idx1] * mask[0]

    return new_x_u_w, new_x_u_s, new_label_u, new_logits_u