# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
from core.libs import set_logger
from config.config import cfg

logger = set_logger()


def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class ModuleHelper:
    @staticmethod
    def BNReLU(num_features, inplace=True):
        return nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=inplace)
        )

    @staticmethod
    def BatchNorm2d(num_features):
        return nn.BatchNorm2d(num_features)

    @staticmethod
    def Conv3x3_BNReLU(in_channels, out_channels, stride=1, dilation=1, groups=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                      groups=groups, bias=False),
            ModuleHelper.BNReLU(out_channels)
        )

    @staticmethod
    def Conv1x1_BNReLU(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            ModuleHelper.BNReLU(out_channels)
        )

    @staticmethod
    def Conv1x1(in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size=1)


class Conv3x3(nn.Module):
    def __init__(self, in_chs, out_chs, dilation=1, dropout=None):
        super(Conv3x3, self).__init__()

        if dropout is None:
            self.conv = ModuleHelper.Conv3x3_BNReLU(in_channels=in_chs, out_channels=out_chs, dilation=dilation)
        else:
            self.conv = nn.Sequential(
                ModuleHelper.Conv3x3_BNReLU(in_channels=in_chs, out_channels=out_chs, dilation=dilation),
                nn.Dropout(dropout)
            )

        initialize_weights(self.conv)

    def forward(self, x):
        return self.conv(x)


class Conv1x1(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(Conv1x1, self).__init__()

        self.conv = ModuleHelper.Conv1x1_BNReLU(in_chs, out_chs)

        initialize_weights(self.conv)

    def forward(self, x):
        return self.conv(x)


# depthwise separable convolution
class DepSepConvolutions(nn.Module):
    def __init__(self, in_chs, out_chs, dilation=1):
        super(DepSepConvolutions, self).__init__()

        # depth wise
        self.DW = ModuleHelper.Conv3x3_BNReLU(in_channels=in_chs, out_channels=in_chs, dilation=dilation, groups=in_chs)
        # point wise
        self.PW = ModuleHelper.Conv1x1_BNReLU(in_channels=in_chs, out_channels=out_chs)

        initialize_weights(self.DW, self.PW)

    def forward(self, x):
        y = self.DW(x)
        y = self.PW(y)

        return y


class DecoderBlock(nn.Module):
    def __init__(self, in_chs, out_chs, dropout=0.1):
        super(DecoderBlock, self).__init__()

        self.doubel_conv = nn.Sequential(
            Conv3x3(in_chs=in_chs, out_chs=out_chs, dropout=dropout),
            Conv3x3(in_chs=out_chs, out_chs=out_chs, dropout=dropout)
        )

        initialize_weights(self.doubel_conv)

    def forward(self, x):
        out = self.doubel_conv(x)
        return out


class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super(Classifier, self).__init__()

        self.conv = nn.Conv2d(in_channels, cfg.MODEL.NUM_CLASSES, kernel_size=1)

        initialize_weights(self.conv)

    def forward(self, x):
        return self.conv(x)


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


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


# position attention
class PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


def Upsample(x, size):
    return nn.functional.interpolate(x, size, mode='bilinear', align_corners=True)


class GateUnit(nn.Module):
    def __init__(self, in_chs):
        super(GateUnit, self).__init__()

        self.conv = nn.Conv2d(in_chs, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        initialize_weights(self.conv)

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)

        return y


class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // 16, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        initialize_weights(self.fc1, self.fc2)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out) * x
        return out


class Aux_Module(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super(Aux_Module, self).__init__()

        self.aux = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        initialize_weights(self.aux)

    def forward(self, x):
        res = self.aux(x)
        return res
