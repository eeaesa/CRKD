# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
import torch.nn.functional as F
from torch.xpu import device


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

    
class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output


class Decoder_URPC(nn.Module):
    def __init__(self, params):
        super(Decoder_URPC, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)
        self.feature_noise = FeatureNoise()

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        if self.training:
            dp3_out_seg = self.out_conv_dp3(Dropout(x, p=0.5))
        else:
            dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        if self.training:
            dp2_out_seg = self.out_conv_dp2(FeatureDropout(x))
        else:
            dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        if self.training:
            dp1_out_seg = self.out_conv_dp1(self.feature_noise(x))
        else:
            dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)
    return x


def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output

class Decoder_feats(nn.Module):
    def __init__(self, params):
        super(Decoder_feats, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

        ft_chns = self.ft_chns[0] + self.ft_chns[1] + self.ft_chns[2] + self.ft_chns[3]
        self.feats_conv = nn.Conv2d(ft_chns, 256,
                                  kernel_size=3, padding=1)

        self.feats_conv_16 = nn.Conv2d(self.ft_chns[0], 256,
                                       kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x3_1 = self.up1(x4, x3)
        x2_1 = self.up2(x3_1, x2)
        x1_1 = self.up3(x2_1, x1)
        x0_1 = self.up4(x1_1, x0)
        output = self.out_conv(x0_1)

        h, w = x2_1.shape[2], x2_1.shape[3]
        x3_1_up = F.interpolate(x3_1, size=(h, w), mode="bilinear",
                             align_corners=True)
        x1_1_down = F.interpolate(x1_1, size=(h, w), mode="bilinear",
                                align_corners=True)
        x0_1_down = F.interpolate(x0_1, size=(h, w), mode="bilinear",
                                  align_corners=True)
        feats = torch.cat([x3_1_up, x2_1, x1_1_down, x0_1_down], dim=1)
        feats = self.feats_conv(feats)
        #feats = self.feats_conv_16(x0_1_down)

        return output, feats

class UNet_feats(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_feats, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder_feats(params)

    def forward(self, x, need_fp=False):
        feature = self.encoder(x)

        if need_fp:
            outs = self.decoder([torch.cat((feat, nn.Dropout2d(0.5)(feat))) for feat in feature])
            return outs.chunk(2)

        output, feats = self.decoder(feature)
        return output, feats

class UNet_CCT(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_CCT, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)
        self.aux_decoder1 = Decoder(params)
        self.aux_decoder2 = Decoder(params)
        self.aux_decoder3 = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        main_seg = self.main_decoder(feature)
        aux1_feature = [FeatureNoise()(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)
        aux2_feature = [Dropout(i) for i in feature]
        aux_seg2 = self.aux_decoder2(aux2_feature)
        aux3_feature = [FeatureDropout(i) for i in feature]
        aux_seg3 = self.aux_decoder3(aux3_feature)
        return main_seg, aux_seg1, aux_seg2, aux_seg3


class UNet_URPC(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_URPC, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder_URPC(params)

    def forward(self, x):
        shape = x.shape[2:]
        feature = self.encoder(x)
        dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg = self.decoder(
            feature, shape)
        return dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg


class UNet_fp(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_fp, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x, need_fp=False):
        feature = self.encoder(x)

        if need_fp:
            outs = self.decoder([torch.cat((feat, nn.Dropout2d(0.5)(feat))) for feat in feature])
            return outs.chunk(2)

        output = self.decoder(feature)
        return output


class Decoder_KD(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

        self.feats_conv_16 = nn.Conv2d(self.ft_chns[0], 256,
                                       kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x3_1 = self.up1(x4, x3)
        x2_1 = self.up2(x3_1, x2)
        x1_1 = self.up3(x2_1, x1)
        x0_1 = self.up4(x1_1, x0)

        output = self.out_conv(x0_1)

        return output, x0_1



class UNet_KD(nn.Module):
    def __init__(self, in_chns, class_num):
        super().__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder_KD(params)

    def forward(self, x, need_fp=False):
        feature = self.encoder(x)

        if need_fp:
            output, feats = self.decoder([nn.Dropout2d(0.5)(feat) for feat in feature])
            return output, feats

        output, feats = self.decoder(feature)

        return output, feats

class UNet_KD_dropout(nn.Module):
    def __init__(self,
                 in_chns, class_num,
                 p=0.5,):
        super().__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder_KD(params)
        self.p = p

    def forward(self, x, need_fp=False):
        feature = self.encoder(x)

        if need_fp:
            output, feats = self.decoder([nn.Dropout2d(self.p)(feat) for feat in feature])
            return output, feats

        output, feats = self.decoder(feature)

        return output, feats

class UNet_KD_new_perbur(nn.Module):
    def __init__(self, in_chns, class_num):
        super().__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder_KD(params)

        self.perbur = FeaturePerturbation()

    def forward(self, x, need_fp=False):
        feature = self.encoder(x)

        if need_fp:
            # output, feats = self.decoder([nn.Dropout2d(0.5)(feat) for feat in feature])
            output, feats = self.decoder([
                (self.perbur(feat)) for feat in feature]
            )
            return output, feats

        output, feats = self.decoder(feature)

        return output, feats

# CrossMatch
class PertDropout(nn.Module):
    def __init__(self, p=0.5):
        super(PertDropout, self).__init__()
        self.p = p
        self.dropouts = [
            nn.Dropout2d(p * 0.5).cuda(),  # Weak
            nn.Dropout2d(p * 1.5).cuda(),  # Strong
        ]
        self.len = len(self.dropouts)

    def __len__(self):
        return self.len

    def forward(self, x):
        rst = []
        for pert_dropout in self.dropouts:
            single_type = []
            for i, feat in enumerate(x):
                perted = pert_dropout(feat)
                single_type.append(perted)
            rst.append(single_type)
        return rst

class UNet_CrossMatch(nn.Module):
    def __init__(self, in_chns, class_num):
        super().__init__()

        params = {
            "in_chns": in_chns,
            "feature_chns": [16, 32, 64, 128, 256],
            "dropout": [0.05, 0.1, 0.2, 0.3, 0.5],
            "class_num": class_num,
            "bilinear": False,
            "acti_func": "relu",
        }

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.ema_decoder = None

        sparse_init_weight(self.encoder)
        sparse_init_weight(self.decoder)
        if self.ema_decoder is not None:
            sparse_init_weight(self.ema_decoder)

        self.pert = PertDropout(0.5).cuda()

    def forward(self, x, need_fp=False, need_ema=False, both=False, drop_rate=0.5):
        feature = self.encoder(x)

        if need_fp:
            features_x = []
            features_u = []
            for feats in feature:
                fx, fu = feats.chunk(2)
                features_x.append(fx)
                features_u.append(fu)

            perted_fus = self.pert(features_u)
            all_zip = zip(features_x, features_u, *perted_fus)
            outs = self.decoder([torch.cat(feats_all) for feats_all in all_zip])
            return outs.chunk(2 + len(self.pert))

        if need_ema:
            pert = (
                nn.FeatureAlphaDropout(0.5)
                if random.random() < 0.5
                else nn.AlphaDropout(0.5)
            )
            return self.decoder(feature), self.ema_decoder(
                [pert(feat) for feat in feature]
            )

        output = self.decoder(feature)
        return output

#---------------feature perturbation--------------------

import torch
import torch.nn as nn

class myFeaturePerturbation(nn.Module):
    def __init__(self, lam=0.9, eps=1e-5):
        """
        向量化实现的特征扰动模块（基于标准差的扰动）
        参数:
            lam (float): 混合权重，控制扰动强度（0 ≤ λ ≤ 1）
            eps (float): 防止除零的小常数
        """
        super().__init__()
        self.lam = lam
        self.eps = eps

    def forward(self, f):
        """
        输入:
            f: 形状为 [B, C, H, W] 的特征图
        输出:
            f_perturbed: 扰动后的特征图 [B, C, H, W]
        """
        B, C, H, W = f.shape

        # 1. 计算每个通道的均值和标准差 (通道级统计量)
        # 将 HxW 展平，保留 batch 和 channel 维度
        f_flat = f.view(B, C, -1)  # [B, C, H*W]

        # 计算通道级均值和标准差
        u_c = f_flat.mean(dim=2, keepdim=True)  # [B, C, 1]
        std_c = f_flat.std(dim=2, keepdim=True)  # [B, C, 1]

        # 2. 计算每个样本的全局均值和标准差 (样本级统计量)
        # 合并通道、H、W 维度，计算每个样本的全局均值和标准差
        f_global = f.view(B, -1)  # [B, C*H*W]
        u_global = f_global.mean(dim=1, keepdim=True).view(B, 1, 1, 1)  # [B, 1, 1, 1]
        std_global = f_global.std(dim=1, keepdim=True).view(B, 1, 1, 1)  # [B, 1, 1, 1]

        # 3. 混合均值和标准差 (广播机制自动对齐维度)
        u_new = self.lam * u_c.view(B, C, 1, 1) + (1 - self.lam) * u_global
        std_new = self.lam * std_c.view(B, C, 1, 1) + (1 - self.lam) * std_global

        # 4. 通道级归一化（使用标准差）
        f_norm = (f - u_c.view(B, C, 1, 1)) / (std_c.view(B, C, 1, 1) + self.eps)

        # 5. 应用新的均值和标准差进行扰动
        f_perturbed = f_norm * (std_new + self.eps) + u_new

        return f_perturbed

class FeaturePerturbation(nn.Module):
    '''
    https://github.com/youngyzzZ/SSL-w2sPC/blob/main/src/networks/feature_perturbe.py
    '''
    def __init__(self, lam=0.9, kap=0.2, eps=1e-6, use_gpu=True):
        super(FeaturePerturbation, self).__init__()
        # self.num_features = num_features
        self.eps = eps
        self.lam = lam
        self.kap = kap
        self.use_gpu = use_gpu

    def forward(self, x):
        device = x.device
        # normalization
        mu = x.mean(dim=[2, 3], keepdim=True)  # [B,C,1,1]
        var = x.var(dim=[2, 3], keepdim=True)  # [B,C,1,1]
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig
        batch_mu = mu.mean(dim=[0], keepdim=True)  # [1,C,1,1]
        batch_psi = (mu.var(dim=[0], keepdim=True) +
                     self.eps).sqrt()  # [1,C,1,1]
        batch_sig = sig.mean(dim=[0], keepdim=True)  # [1,C,1,1]
        batch_phi = (sig.var(dim=[0], keepdim=True) +
                     self.eps).sqrt()  # [1,C,1,1]
        epsilon = torch.empty(1).uniform_(-self.kap, self.kap).to(device)
        gamma = self.lam * sig + (1 - self.lam) * \
            batch_sig + epsilon * batch_phi
        beta = self.lam * mu + (1 - self.lam) * batch_mu + epsilon * batch_psi
        x_aug = gamma * x_normed + beta
        return x_aug


if __name__ == '__main__':
    module = FeaturePerturbation().cuda()
    inputs = torch.randn(4, 128, 256, 256)
    outpust = module(inputs)
    print(outpust.size())

# if __name__ == "__main__":
#     # check_vssm_equals_vmambadp()
#     model = UNet_CrossMatch(class_num=4, in_chns=1).to('cuda') # 2.29 GMac 1.81 M
#     int = torch.randn(2,1,224,224).cuda()
#     out = model(int, need_fp=True)
#     # print(out.shape)
#
#     from ptflops import get_model_complexity_info
#
#     flops, params = get_model_complexity_info(model, (1, 224, 224), as_strings=True, print_per_layer_stat=True)
#     print("%s %s" % (flops, params))