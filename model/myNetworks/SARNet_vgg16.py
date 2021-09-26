import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.VGG.VGG import Backbone_VGG16_in3

from model.modules.model_libs import Classifier, Upsample, Aux_Module, Conv3x3, ModuleHelper, DecoderBlock, \
    ChannelAttention, SpatialAttention
from model.modules.spatial_ocr import SpatialGatherModule, SpatialOCR_Module


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SARNet_v(nn.Module):
    def __init__(self, channel=32):
        super(SARNet_v, self).__init__()
        # ---- VGG16 Backbone ----
        (
            self.encoder1,
            self.encoder2,
            self.encoder4,
            self.encoder8,
            self.encoder16,
        ) = Backbone_VGG16_in3()

        # Lateral layers
        reduction_dim = 256
        self.lateral_conv0 = Conv3x3(64, 64)
        self.lateral_conv1 = Conv3x3(128, reduction_dim)
        self.lateral_conv2 = Conv3x3(256, reduction_dim, dropout=0.1)
        self.lateral_conv3 = Conv3x3(512, reduction_dim, dropout=0.1)
        self.lateral_conv4 = Conv3x3(512, reduction_dim, dropout=0.1)

        self.conv3x3_ocr = ModuleHelper.Conv3x3_BNReLU(768, 256)
        self.ocr_aux = nn.Sequential(
            ModuleHelper.Conv1x1_BNReLU(reduction_dim * 3, reduction_dim),
            Classifier(reduction_dim, num_classes=1)
        )
        self.cls = Classifier(reduction_dim, num_classes=1)

        self.ocr_gather_head = SpatialGatherModule(cls_num=1)
        self.ocr_distri_head = SpatialOCR_Module(in_channels=256, key_channels=256, out_channels=256, scale=1,
                                                 dropout=0.05)

        # 5-->4
        self.CAM_5_0 = ChannelAttention(in_planes=reduction_dim)
        self.CAM_5_1 = ChannelAttention(in_planes=reduction_dim)
        self.SAM_5 = SpatialAttention()

        self.gamma_5 = nn.Parameter(torch.ones(1))
        self.gate_5 = nn.Sequential(
            nn.Conv2d(256 + 256, 1, 1),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.conv_out_5 = DecoderBlock(2, 1, dropout=0.0)

        # 4-->3
        self.CAM_4_0 = ChannelAttention(in_planes=reduction_dim)
        self.CAM_4_1 = ChannelAttention(in_planes=reduction_dim)
        self.SAM_4 = SpatialAttention()
        self.gamma_4 = nn.Parameter(torch.ones(1))
        self.gate_4 = nn.Sequential(
            nn.Conv2d(256 + 256, 1, 1),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.conv_out_4 = DecoderBlock(2, 1, dropout=0.0)

        # 3-->2
        self.CAM_3_0 = ChannelAttention(in_planes=reduction_dim)
        self.CAM_3_1 = ChannelAttention(in_planes=reduction_dim)
        self.SAM_3 = SpatialAttention()
        self.gamma_3 = nn.Parameter(torch.ones(1))
        self.gate_3 = nn.Sequential(
            nn.Conv2d(256 + 256, 1, 1),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.conv_out_3 = DecoderBlock(2, 1, dropout=0.0)

        # 2-->1
        self.CAM_2_0 = ChannelAttention(in_planes=reduction_dim)
        self.CAM_2_1 = ChannelAttention(in_planes=reduction_dim)
        self.SAM_2 = SpatialAttention()
        self.gamma_2 = nn.Parameter(torch.ones(1))
        self.gate_2 = nn.Sequential(
            nn.Conv2d(256 + 256, 1, 1),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.conv_out_3 = DecoderBlock(2, 1, dropout=0.0)

        # 1-->0
        self.CAM_1_0 = ChannelAttention(in_planes=reduction_dim)
        self.CAM_1_1 = ChannelAttention(in_planes=64)
        self.SAM_1 = SpatialAttention()
        self.gamma_1 = nn.Parameter(torch.ones(1))
        self.gate_1 = nn.Sequential(
            nn.Conv2d(256 + 64, 1, 1),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.conv_out_3 = DecoderBlock(2, 1, dropout=0.0)

        self.ra4_conv1 = BasicConv2d(256, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)

        self.ra3_conv1 = BasicConv2d(256, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

        self.ra2_conv1 = BasicConv2d(256, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

        self.ra1_conv1 = BasicConv2d(256, 64, kernel_size=1)
        self.ra1_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra1_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra1_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

        self.ra0_conv1 = BasicConv2d(64, 64, kernel_size=1)
        self.ra0_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra0_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra0_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x0 = self.encoder1(x)
        x1 = self.encoder2(x0)
        x2 = self.encoder4(x1)
        x3 = self.encoder8(x2)
        x4 = self.encoder16(x3)

        lateral_x0 = self.lateral_conv0(x0)
        lateral_x1 = self.lateral_conv1(x1)
        lateral_x2 = self.lateral_conv2(x2)
        lateral_x3 = self.lateral_conv3(x3)
        lateral_x4 = self.lateral_conv4(x4)

        lateral_x3 = F.interpolate(lateral_x3, scale_factor=2, mode='bilinear')
        lateral_x4 = F.interpolate(lateral_x4, scale_factor=4, mode='bilinear')

        out_cat = torch.cat((lateral_x2, lateral_x3, lateral_x4), 1)

        out_aux = self.ocr_aux(out_cat)

        feats = self.conv3x3_ocr(out_cat)
        context = self.ocr_gather_head(feats, out_aux)
        g_feats = self.ocr_distri_head(feats, context)

        ra5_feat = self.cls(g_feats)

        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=4,
                                      mode='bilinear')

        ### 5-->4 ###
        x = -1 * (torch.sigmoid(ra5_feat)) + 1
        g_feats_temp = g_feats.mul(self.CAM_5_0(g_feats))
        lateral_x4_temp = lateral_x4.mul(self.CAM_5_1(lateral_x4))
        out_cat_cam_5 = torch.cat((lateral_x4_temp, g_feats_temp), 1)
        lateral_gate_x4 = self.gamma_5 * self.gate_5(out_cat_cam_5) * lateral_x4_temp
        x = x.expand(-1, 256, -1, -1).mul(lateral_gate_x4)
        x = self.ra4_conv1(x)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        ra4_feat = self.ra4_conv5(x)

        refine_out_4 = ra4_feat + ra5_feat
        lateral_map_4 = F.interpolate(refine_out_4, scale_factor=4, mode='bilinear')

        ### 4-->3 ###
        x = -1 * (torch.sigmoid(refine_out_4)) + 1
        lateral_x4_temp = lateral_x4.mul(self.CAM_4_0(lateral_x4))
        lateral_x3_temp = lateral_x3.mul(self.CAM_4_1(lateral_x3))

        out_cat_cam_3 = torch.cat((lateral_x3, lateral_x4_temp), 1)
        lateral_gate_x3 = self.gamma_4 * self.gate_4(out_cat_cam_3) * lateral_x3_temp
        x = x.expand(-1, 256, -1, -1).mul(lateral_gate_x3)
        x = self.ra3_conv1(x)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)

        refine_out_3 = ra3_feat + refine_out_4
        lateral_map_3 = F.interpolate(refine_out_3, scale_factor=4, mode='bilinear')

        ### 3-->2 ###
        x = -1 * (torch.sigmoid(refine_out_3)) + 1
        lateral_x3_temp = lateral_x3.mul(self.CAM_3_0(lateral_x3))
        lateral_x2_temp = lateral_x2.mul(self.CAM_3_1(lateral_x2))
        out_cat_cam_2 = torch.cat((lateral_x2, lateral_x3_temp), 1)
        lateral_gate_x2 = self.gamma_3 * self.gate_3(out_cat_cam_2) * lateral_x2_temp
        x = x.expand(-1, 256, -1, -1).mul(lateral_gate_x2)
        x = self.ra2_conv1(x)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(x)
        refine_out_2 = ra2_feat + refine_out_3  # bs,1,88, 88
        lateral_map_2 = F.interpolate(refine_out_2, scale_factor=4, mode='bilinear')

        ### 2-->1 ###
        refine_crop_1 = F.interpolate(refine_out_2, scale_factor=2, mode='bilinear')
        x2_crop_1 = F.interpolate(lateral_x2, scale_factor=2, mode='bilinear')
        x = -1 * (torch.sigmoid(refine_crop_1)) + 1  # bs,1,176, 176
        lateral_x2_temp = x2_crop_1.mul(self.CAM_2_0(x2_crop_1))
        lateral_x1_temp = lateral_x1.mul(self.CAM_2_1(lateral_x1))
        out_cat_cam_1 = torch.cat((lateral_x2_temp, lateral_x1_temp), 1)
        lateral_gate_x1 = self.gamma_2 * self.gate_2(out_cat_cam_1) * lateral_x1_temp

        x = x.expand(-1, 256, -1, -1).mul(lateral_gate_x1)
        x = self.ra1_conv1(x)
        x = F.relu(self.ra1_conv2(x))
        x = F.relu(self.ra1_conv3(x))
        ra1_feat = self.ra1_conv4(x)
        refine_out_1 = ra1_feat + refine_crop_1  # bs,1,176, 176
        lateral_map_1 = F.interpolate(refine_out_1, scale_factor=2, mode='bilinear')

        ### 1-->0 ###
        refine_crop_0 = F.interpolate(refine_out_1, scale_factor=2, mode='bilinear')
        x1_crop_1 = F.interpolate(lateral_x1, scale_factor=2, mode='bilinear')
        x = -1 * (torch.sigmoid(refine_crop_0)) + 1
        lateral_x1_temp = x1_crop_1.mul(self.CAM_1_0(x1_crop_1))
        lateral_x0_temp = lateral_x0.mul(self.CAM_1_1(lateral_x0))
        out_cat_cam_0 = torch.cat((lateral_x1_temp, lateral_x0_temp), 1)
        lateral_gate_x0 = self.gamma_1 * self.gate_1(out_cat_cam_0) * lateral_x0_temp

        x = x.expand(-1, 64, -1, -1).mul(lateral_gate_x0)
        x = self.ra0_conv1(x)
        x = F.relu(self.ra0_conv2(x))
        x = F.relu(self.ra0_conv3(x))
        ra0_feat = self.ra0_conv4(x)
        refine_out_0 = ra0_feat + refine_crop_0
        lateral_map_0 = F.interpolate(refine_out_0, scale_factor=1, mode='bilinear')

        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1, lateral_map_0
