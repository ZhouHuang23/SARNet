"""
  Cross-level Gating Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules.model_libs import Conv1x1, DecoderBlock, initialize_weights
import math


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        y = self.sigmoid(out) * x
        return y


class AlignModule(nn.Module):
    def __init__(self, inplane, outplane):
        super(AlignModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane * 2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, low_feature, h_feature):
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=False)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    @staticmethod
    def flow_warp(input, flow, size):
        out_h, out_w = size  # 对应高分辨率的low-level feature的特征图尺寸
        n, c, h, w = input.size()  # 对应低分辨率的high-level feature的4个输入维度

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # 从-1到1等距离生成out_h个点，每一行重复out_w个点，最终生成(out_h, out_w)的像素点
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        # 生成w的转置矩阵
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        # 展开后进行合并
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        # grid指定由input空间维度归一化的采样像素位置，其大部分值应该在[ -1, 1]的范围内
        # 如x=-1,y=-1是input的左上角像素，x=1,y=1是input的右下角像素。
        # 具体可以参考《Spatial Transformer Networks》，下方参考文献[2]
        output = F.grid_sample(input, grid, align_corners=True)
        return output


class ECAModule(nn.Module):
    """Efficient Channel Attention.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel):
        super(ECAModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        k_size = self._get_k_value(channel)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _get_k_value(C, gamma=2, b=1):
        t = int(abs(math.log(C, 2) + b) / gamma)
        k = t if t % 2 else t + 1
        return k

    def forward(self, x):
        # x: input features with shape [b, c, h, w]

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        out = x * y.expand_as(x)

        return out


class CLGD(nn.Module):
    """
    Cross-level Gating Decoder
    """

    def __init__(self, in_chs_l, in_chs_h, in_chs_g, out_chs, reduction=False, add_global_f=False, dropout=0.0):
        super(CLGD, self).__init__()

        self.add_global_f = add_global_f
        self.reduction = reduction

        if reduction:
            self.high_feat_conv = Conv1x1(in_chs_h, in_chs_l)
            initialize_weights(self.high_feat_conv)
        self.align = AlignModule(in_chs_l, in_chs_l)

        self.cam_low_f = ChannelAttention(in_chs_l)
        self.gamma = nn.Parameter(torch.ones(1))
        self.gate = nn.Sequential(
            nn.Conv2d(2 * in_chs_l, 1, 1),
            nn.Sigmoid()
        )  # att

        if add_global_f:
            self.cam_global_f = ChannelAttention(in_chs_g)
            self.conv_out = DecoderBlock(2 * in_chs_l + in_chs_g, out_chs, dropout=dropout)
            self.conv_g = None
            if in_chs_l != in_chs_g:
                self.conv_g = Conv1x1(in_chs_g, in_chs_l)
                self.cam_global_f = ChannelAttention(in_chs_l)
                self.conv_out = DecoderBlock(2 * in_chs_l + in_chs_l, out_chs, dropout=dropout)
                initialize_weights(self.conv_g)

            initialize_weights(self.cam_global_f)
        else:
            self.conv_out = DecoderBlock(2 * in_chs_l, out_chs, dropout=dropout)

        initialize_weights(self.conv_out, self.gate, self.align, self.cam_low_f)

    def forward(self, low_feat, high_feat, global_f):
        """
            inputs :
                x : low level feature(N,C,H,W)  y:high level feature(N,C,H,W)
            returns :
                out :  cross-level gating decoder feature
        """

        if self.reduction:
            high_feat = self.high_feat_conv(high_feat)

        high_feat = self.align(low_feat, high_feat)

        # low feat refine
        low_feat = self.cam_low_f(low_feat)
        x = torch.cat([low_feat, high_feat], dim=1)
        low_feat = self.gamma * self.gate(x) * low_feat

        if self.add_global_f:
            if self.conv_g is not None:
                global_f = self.conv_g(global_f)
            global_f = self.cam_global_f(global_f)
            global_f = nn.functional.interpolate(global_f, size=low_feat.size()[2:], mode='bilinear',
                                                 align_corners=True)
            x = torch.cat([low_feat, high_feat, global_f], 1)
        else:
            x = torch.cat([low_feat, high_feat], 1)

        y = self.conv_out(x)

        return y


if __name__ == '__main__':
    from torchstat import stat

    model = CLGD(in_chs_l=256, in_chs_h=256, in_chs_g=256, out_chs=256, dropout=0.1)
    stat(model, (3, 512, 512))