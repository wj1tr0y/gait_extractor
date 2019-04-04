# coding: utf-8
"""
Created on Jan 11, 2018

@author: guoweiyu
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1, padding=1, group=1, bn=False, act=True):
    """3x3 convolution with padding"""
    layers = [nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=padding, groups=group, bias=False)]
    if bn:
        layers.append(nn.BatchNorm2d(out_planes))
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def conv_dw(inp, oup, stride, pad=1, gfactor=1, bn=True, act=True):
    layers = [nn.Conv2d(inp, inp, 3, stride, pad, groups=inp // gfactor, bias=False)]
    if bn:
        layers.append(nn.BatchNorm2d(inp))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(inp, oup, 1, 1, 0, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(oup))
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def conv1x1(in_planes, out_planes, pad=0, group=1, bn=False, act=True):
    """1x1 convolution with padding"""
    layers = [nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                        padding=pad, groups=group, bias=False)]
    if bn:
        layers.append(nn.BatchNorm2d(out_planes))
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def deconv(in_planes, out_planes, kernel_shape=3, stride_shape=2, pad_shape=1, group=1, bn=False):
    layers = [
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_shape, stride=stride_shape, padding=pad_shape,
                           groups=group, bias=False)]
    if bn:
        layers.append(nn.BatchNorm2d(out_planes))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class SegNet(nn.Module):
    def __init__(self, input_chs=3, filter_num=64):
        super(SegNet, self).__init__()
        self.bn = True
        # encoder  
        self.conv1 = conv3x3(input_chs, filter_num, stride=1, padding=1, group=1, bn=self.bn)
        self.conv2 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv3 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv4 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv5 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv6 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv7 = conv3x3(filter_num, filter_num, stride=1, padding=0, group=1, bn=self.bn)

        n_cat_filter = filter_num * 2
        # decoder
        self.deconv1 = deconv(filter_num, filter_num, kernel_shape=3, stride_shape=1, pad_shape=0, group=1,
                              bn=self.bn)
        self.deconv2 = deconv(n_cat_filter, filter_num, kernel_shape=(4, 3), stride_shape=2, pad_shape=1, group=1,
                              bn=self.bn)
        self.deconv3 = deconv(n_cat_filter, filter_num, kernel_shape=3, stride_shape=2, pad_shape=1, group=1,
                              bn=self.bn)
        self.deconv4 = deconv(n_cat_filter, filter_num, kernel_shape=(4, 3), stride_shape=2, pad_shape=1, group=1,
                              bn=self.bn)
        self.deconv5 = deconv(n_cat_filter, filter_num, kernel_shape=(3, 4), stride_shape=2, pad_shape=1, group=1,
                              bn=self.bn)
        self.deconv6 = deconv(n_cat_filter, filter_num, kernel_shape=(4, 4), stride_shape=2, pad_shape=1, group=1,
                              bn=self.bn)

        # fusion
        self.conv_fusion6 = conv1x1(filter_num, filter_num, bn=self.bn)
        self.seg_score = conv1x1(filter_num, 1, bn=self.bn, act=False)

    def forward(self, x):
        x = self.conv1(x)
        res_conv2 = self.conv2(x)
        res_conv3 = self.conv3(res_conv2)
        res_conv4 = self.conv4(res_conv3)
        res_conv5 = self.conv5(res_conv4)
        res_conv6 = self.conv6(res_conv5)

        x = self.conv7(res_conv6)
        x = self.deconv1(x)

        x = torch.cat((res_conv6, x), 1)
        res_deconv2 = self.deconv2(x)
        x = torch.cat((res_conv5, res_deconv2), 1)

        res_deconv3 = self.deconv3(x)
        x = torch.cat((res_conv4, res_deconv3), 1)

        res_deconv4 = self.deconv4(x)
        x = torch.cat((res_conv3, res_deconv4), 1)

        res_deconv5 = self.deconv5(x)
        x = torch.cat((res_conv2, res_deconv5), 1)
        x = self.deconv6(x)

        x = self.conv_fusion6(x)
        x = F.sigmoid(self.seg_score(x))
        return x


class SegNetHalf(nn.Module):
    def __init__(self, input_chs=3, filter_num=64):
        super(SegNetHalf, self).__init__()
        self.bn = True
        # encoder  
        self.conv1 = conv3x3(input_chs, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv2 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv3 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv4 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv5 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv6 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        # self.conv7 = conv3x3(filter_num, filter_num, stride=1, padding=0, group=1, bn=self.bn)

        n_cat_filter = filter_num * 2
        # decoder
        self.deconv1 = deconv(filter_num, filter_num, kernel_shape=3, stride_shape=1, pad_shape=0, group=1,
                              bn=self.bn)
        self.deconv2 = deconv(n_cat_filter, filter_num, kernel_shape=(4, 3), stride_shape=2, pad_shape=1, group=1,
                              bn=self.bn)
        self.deconv3 = deconv(n_cat_filter, filter_num, kernel_shape=3, stride_shape=2, pad_shape=1, group=1,
                              bn=self.bn)
        self.deconv4 = deconv(n_cat_filter, filter_num, kernel_shape=(4, 3), stride_shape=2, pad_shape=1, group=1,
                              bn=self.bn)
        self.deconv5 = deconv(n_cat_filter, filter_num, kernel_shape=(3, 4), stride_shape=2, pad_shape=1, group=1,
                              bn=self.bn)
        self.deconv6 = deconv(n_cat_filter, filter_num, kernel_shape=(4, 4), stride_shape=2, pad_shape=1, group=1,
                              bn=self.bn)
        # fusion
        # self.conv_fusion6 = conv1x1(filter_num, filter_num, bn=self.bn)
        self.seg_score = conv1x1(filter_num, 1, bn=self.bn, act=False)

    def forward(self, x):
        res_conv1 = self.conv1(x)
        res_conv2 = self.conv2(res_conv1)
        res_conv3 = self.conv3(res_conv2)
        res_conv4 = self.conv4(res_conv3)
        res_conv5 = self.conv5(res_conv4)

        x = self.conv6(res_conv5)
        x = self.deconv1(x)

        x = torch.cat((res_conv5, x), 1)  # fu2
        res_deconv2 = self.deconv2(x)

        x = torch.cat((res_conv4, res_deconv2), 1)  # fu3
        res_deconv3 = self.deconv4(x)

        x = torch.cat((res_conv3, res_deconv3), 1)  # fu4
        res_deconv4 = self.deconv4(x)

        x = torch.cat((res_conv2, res_deconv4), 1)  # fu5
        res_deconv5 = self.deconv5(x)

        x = torch.cat((res_conv1, res_deconv5), 1)  # fu1
        x = self.deconv6(x)

        x = F.sigmoid(self.seg_score(x))
        return x


class SegMobileNet(nn.Module):
    def __init__(self, input_chs=3, filter_num=64):
        super(SegMobileNet, self).__init__()
        self.bn = True
        self.gf = 2

        def deconv_dw(inp, oup, kernel_size, stride, pad=1, bn=True, gfactor=1, ):
            layers = [nn.ConvTranspose2d(inp, oup, kernel_size, stride, pad, groups=oup / gfactor, bias=False)]
            if bn:
                layers.append(nn.BatchNorm2d(oup))
            layers.append(nn.ReLU(inplace=True))

            layers.append(nn.Conv2d(oup, oup, 1, 1, 0, bias=False))
            if bn:
                layers.append(nn.BatchNorm2d(oup))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        # encoder   
        self.conv1 = conv3x3(input_chs, filter_num, stride=1, padding=1, group=1, bn=self.bn)
        self.conv2 = conv_dw(filter_num, filter_num, stride=2, pad=1, bn=self.bn, gfactor=self.gf)
        self.conv3 = conv_dw(filter_num, filter_num, stride=2, pad=1, bn=self.bn, gfactor=self.gf)
        self.conv4 = conv_dw(filter_num, filter_num, stride=2, pad=1, bn=self.bn, gfactor=self.gf)
        self.conv5 = conv_dw(filter_num, filter_num, stride=2, pad=1, bn=self.bn, gfactor=self.gf)

        self.conv6 = conv_dw(filter_num, filter_num, stride=2, pad=1, bn=self.bn, gfactor=self.gf)
        self.conv7 = conv_dw(filter_num, filter_num, stride=1, pad=0, bn=self.bn, gfactor=self.gf)

        n_cat_filter = filter_num * 2
        # decoder
        self.deconv1 = deconv_dw(filter_num, filter_num, kernel_size=3, stride=1, pad=0, bn=self.bn, gfactor=self.gf)
        self.deconv2 = deconv_dw(n_cat_filter, filter_num, kernel_size=(4, 3), stride=2, pad=1, bn=self.bn,
                                 gfactor=self.gf)
        self.deconv3 = deconv_dw(n_cat_filter, filter_num, kernel_size=3, stride=2, pad=1, bn=self.bn, gfactor=self.gf)
        self.deconv4 = deconv_dw(n_cat_filter, filter_num, kernel_size=(4, 3), stride=2, pad=1, bn=self.bn,
                                 gfactor=self.gf)
        self.deconv5 = deconv_dw(n_cat_filter, filter_num, kernel_size=(3, 4), stride=2, pad=1, bn=self.bn,
                                 gfactor=self.gf)
        self.deconv6 = deconv_dw(n_cat_filter, filter_num, kernel_size=(4, 4), stride=2, pad=1, bn=self.bn,
                                 gfactor=self.gf)

        # fusion
        self.conv_fusion6 = conv1x1(filter_num, filter_num, bn=self.bn)
        self.seg_score = conv1x1(filter_num, 1, bn=False, act=False)

    def forward(self, x):
        x = self.conv1(x)
        res_conv2 = self.conv2(x)
        res_conv3 = self.conv3(res_conv2)
        res_conv4 = self.conv4(res_conv3)
        res_conv5 = self.conv5(res_conv4)
        res_conv6 = self.conv6(res_conv5)

        x = self.conv7(res_conv6)
        x = self.deconv1(x)

        x = torch.cat((res_conv6, x), 1)
        res_deconv2 = self.deconv2(x)
        x = torch.cat((res_conv5, res_deconv2), 1)

        res_deconv3 = self.deconv3(x)
        x = torch.cat((res_conv4, res_deconv3), 1)

        res_deconv4 = self.deconv4(x)
        x = torch.cat((res_conv3, res_deconv4), 1)

        res_deconv5 = self.deconv5(x)
        x = torch.cat((res_conv2, res_deconv5), 1)
        x = F.relu(self.deconv6(x))

        x = self.conv_fusion6(x)
        x = F.sigmoid(self.seg_score(x))
        return x


class SegMobileNetHalf(nn.Module):
    def __init__(self, input_chs=3, filter_num=64):
        super(SegMobileNetHalf, self).__init__()
        self.bn = True
        self.gf = 2

        def deconv_dw(inp, oup, kernel_size, stride, pad=1, bn=True, gfactor=1):
            layers = [nn.ConvTranspose2d(inp, oup, kernel_size, stride, pad, groups=oup / gfactor, bias=False)]
            if bn:
                layers.append(nn.BatchNorm2d(oup))
            layers.append(nn.ReLU(inplace=True))

            layers.append(nn.Conv2d(oup, oup, 1, 1, 0, bias=False))
            if bn:
                layers.append(nn.BatchNorm2d(oup))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        # encoder   
        self.conv1 = conv3x3(input_chs, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv2 = conv_dw(filter_num, filter_num, stride=2, pad=1, bn=self.bn, gfactor=self.gf)
        self.conv3 = conv_dw(filter_num, filter_num, stride=2, pad=1, bn=self.bn, gfactor=self.gf)
        self.conv4 = conv_dw(filter_num, filter_num, stride=2, pad=1, bn=self.bn, gfactor=self.gf)
        self.conv5 = conv_dw(filter_num, filter_num, stride=2, pad=1, bn=self.bn, gfactor=self.gf)
        self.conv6 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        # self.conv7 = conv3x3(filter_num, filter_num, stride=1, padding=0, group=1, bn=False)

        n_cat_filter = filter_num * 2
        # decoder
        self.deconv1 = deconv(filter_num, filter_num, kernel_shape=3, stride_shape=1, pad_shape=0, group=1,
                              bn=self.bn)
        self.deconv3 = deconv_dw(n_cat_filter, filter_num, kernel_size=(4, 3), stride=2, pad=1, bn=self.bn,
                                 gfactor=self.gf)
        self.deconv4 = deconv_dw(n_cat_filter, filter_num, kernel_size=3, stride=2, pad=1, bn=self.bn, gfactor=self.gf)
        self.deconv5 = deconv_dw(n_cat_filter, filter_num, kernel_size=(4, 3), stride=2, pad=1, bn=self.bn,
                                 gfactor=self.gf)
        self.deconv6 = deconv_dw(n_cat_filter, filter_num, kernel_size=(3, 4), stride=2, pad=1, bn=self.bn,
                                 gfactor=self.gf)
        self.deconv7 = deconv(n_cat_filter, filter_num, kernel_shape=(4, 4), stride_shape=2, pad_shape=1, group=1,
                              bn=self.bn)
        # fusion
        # self.conv_fusion6 = conv1x1(filter_num, filter_num, bn=False)
        self.seg_score = conv1x1(filter_num, 1, bn=self.bn, act=False)

    def forward(self, x):
        res_conv1 = self.conv1(x)
        res_conv2 = self.conv2(res_conv1)
        res_conv3 = self.conv3(res_conv2)
        res_conv4 = self.conv4(res_conv3)
        res_conv5 = self.conv5(res_conv4)

        x = self.conv6(res_conv5)
        x = self.deconv1(x)

        x = torch.cat((res_conv5, x), 1)  # fu2
        res_deconv3 = self.deconv3(x)

        x = torch.cat((res_conv4, res_deconv3), 1)  # fu3
        res_deconv4 = self.deconv4(x)

        x = torch.cat((res_conv3, res_deconv4), 1)  # fu4
        res_deconv5 = self.deconv5(x)

        x = torch.cat((res_conv2, res_deconv5), 1)  # fu5
        res_deconv6 = self.deconv6(x)

        x = torch.cat((res_conv1, res_deconv6), 1)  # fu1
        x = self.deconv7(x)

        x = F.sigmoid(self.seg_score(x))
        return x


class SegUpsample(nn.Module):
    def __init__(self, input_chs=3, filter_num=64):
        super(SegUpsample, self).__init__()
        self.bn = True

        def upsample_deconv(input_chs, output_chs, out_size, mode='bilinear'):
            layers = [conv3x3(input_chs, output_chs, stride=1, padding=1, group=1, bn=True, act=False)]
            layers.append(nn.Upsample(size=out_size, mode=mode))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        # encoder
        self.conv1 = conv3x3(input_chs, filter_num, stride=1, padding=1, group=1, bn=self.bn)
        self.conv2 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv3 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv4 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv5 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv6 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv7 = conv3x3(filter_num, filter_num, stride=1, padding=0, group=1, bn=self.bn)

        n_cat_filter = filter_num * 2
        # decoder
        self.upsample_deconv1 = upsample_deconv(filter_num, filter_num, (5, 4))
        # nn.ConvTranspose2d(filter_num, filter_num, kernel_size=3, stride=1, padding=0, groups=1, bias=True)
        self.upsample_deconv2 = upsample_deconv(n_cat_filter, filter_num, (10, 7))
        self.upsample_deconv3 = upsample_deconv(n_cat_filter, filter_num, (19, 13))
        self.upsample_deconv4 = upsample_deconv(n_cat_filter, filter_num, (38, 25))
        self.upsample_deconv5 = upsample_deconv(n_cat_filter, filter_num, (75, 50))
        self.upsample_deconv6 = upsample_deconv(n_cat_filter, filter_num, (150, 100))

        # fusion
        self.conv_fusion6 = conv1x1(filter_num, filter_num, bn=False)
        self.seg_score = conv1x1(filter_num, 1, bn=False, act=False)

    def forward(self, x):
        x = self.conv1(x)
        res_conv2 = self.conv2(x)
        res_conv3 = self.conv3(res_conv2)
        res_conv4 = self.conv4(res_conv3)
        res_conv5 = self.conv5(res_conv4)
        res_conv6 = self.conv6(res_conv5)

        x = self.conv7(res_conv6)
        x = self.upsample_deconv1(x)

        x = torch.cat((res_conv6, x), 1)

        res_deconv2 = self.upsample_deconv2(x)
        x = torch.cat((res_conv5, res_deconv2), 1)

        res_deconv3 = self.upsample_deconv3(x)
        x = torch.cat((res_conv4, res_deconv3), 1)

        res_deconv4 = self.upsample_deconv4(x)
        x = torch.cat((res_conv3, res_deconv4), 1)

        res_deconv5 = self.upsample_deconv5(x)
        x = torch.cat((res_conv2, res_deconv5), 1)

        x = self.upsample_deconv6(x)

        x = self.conv_fusion6(x)
        x = F.sigmoid(self.seg_score(x))
        return x


class DeconvUnit(nn.Module):
    def __init__(self, in_chs, out_chs, in_width, in_height, out_width, out_height, first_groups=4, conv3x3_groups=2):
        # type: (object, object, object, object, object, object) -> object
        super(DeconvUnit, self).__init__()
        self.out_width = out_width
        self.out_height = out_height
        self.in_chs = in_chs
        self.first_groups = first_groups

        out_chs_conv1x1 = out_width * out_height

        self.group_conv1x1 = conv1x1(in_chs, out_chs_conv1x1, bn=True, group=first_groups)
        in_chs_conv3x3 = in_width * in_height

        bottleneck_channels = out_chs // 4
        self.depth_wise_conv3x3 = conv3x3(in_chs_conv3x3, bottleneck_channels, stride=1, padding=1,
                                          group=conv3x3_groups, bn=True)

        self.group_conv1x1_ext = conv1x1(bottleneck_channels, out_chs, bn=True, group=bottleneck_channels)

    def ch_shuffle(self, x, groups):
        batch_size, num_channels, height, width = x.data.size()
        if num_channels % groups == 0:
            channels_per_group = num_channels // groups
            x = x.view(batch_size, groups, channels_per_group, height, width)
            x = torch.transpose(x, 1, 2).contiguous()

        x = x.view(batch_size, -1, self.out_height, self.out_width)
        return x

    def forward(self, x):
        x = self.group_conv1x1(x)
        x = self.ch_shuffle(x, self.first_groups)
        x = self.depth_wise_conv3x3(x)
        x = self.group_conv1x1_ext(x)
        return x


class DeconvUnit2(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, padding, upscale_factor=2):
        # type: (object, object, object, object, object, object) -> object
        super(DeconvUnit2, self).__init__()

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        conv_in_chs = in_chs / (upscale_factor * upscale_factor)
        layers = [nn.Conv2d(conv_in_chs, out_chs, kernel_size=kernel_size, padding=padding, groups=1, bias=False),
                  nn.BatchNorm2d(out_chs),
                  nn.ReLU(inplace=True)
                  ]
        self.conv_bn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.conv_bn(x)
        return x


class DeconvUnitPool(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, padding, upscale_factor=2, typ='max'):
        # type: (object, object, object, object, object, object) -> object
        super(DeconvUnitPool, self).__init__()

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        if typ == 'max':
            self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=padding)
        else:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=padding)

        conv_in_chs = in_chs / (upscale_factor * upscale_factor)
        layers = [nn.Conv2d(conv_in_chs, out_chs, kernel_size=1, padding=0, groups=1,
                            bias=False),
                  nn.BatchNorm2d(out_chs),
                  nn.ReLU(inplace=True)
                  ]
        self.conv_bn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.pool(x)
        x = self.conv_bn(x)
        return x


class DeconvUnitCrop(nn.Module):
    def __init__(self, in_chs, out_chs, out_size, upscale_factor=2):
        # type: (object, object, object, object, object, object) -> object
        super(DeconvUnitCrop, self).__init__()

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        conv_in_chs = in_chs / (upscale_factor * upscale_factor)
        layers = [nn.Conv2d(conv_in_chs, out_chs, kernel_size=1, padding=0, groups=1,
                            bias=False),
                  nn.BatchNorm2d(out_chs),
                  nn.ReLU(inplace=True)
                  ]
        self.conv_bn = nn.Sequential(*layers)
        self.height = out_size[0]
        self.width = out_size[1]

    def _crop(self, x):
        _, _, h, w = x.size()
        pad_l = 0
        pad_r = 0
        pad_t = 0
        pad_b = 0
        if w > self.width:
            pad_l = self.width - w
        if h > self.height:
            pad_t = self.height - h
        return F.pad(x, [pad_l, pad_r, pad_t, pad_b])

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self._crop(x)
        x = self.conv_bn(x)

        return x


class SegShuffle(nn.Module):
    def __init__(self, input_chs=3, filter_num=64, typ='conv'):
        super(SegShuffle, self).__init__()
        self.bn = True
        # nn.PixelShuffle
        # encoder
        self.conv1 = conv3x3(input_chs, filter_num, stride=1, padding=1, group=1, bn=self.bn)
        self.conv2 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv3 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv4 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv5 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv6 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv7 = conv3x3(filter_num, filter_num, stride=1, padding=0, group=1, bn=self.bn)

        # decoder
        n_cat_filter = filter_num * 2
        '''
        self.deconv1 = DeconvUnit(filter_num, filter_num, 2, 3, 4, 5, first_groups=4, conv3x3_groups=2)
        self.deconv2 = DeconvUnit(n_cat_filter, filter_num, 4, 5, 7, 10, first_groups=2, conv3x3_groups=4)
        self.deconv3 = DeconvUnit(n_cat_filter, filter_num, 7, 10, 13, 19, first_groups=1, conv3x3_groups=2)
        self.deconv4 = DeconvUnit(n_cat_filter, filter_num, 13, 19, 25, 38, first_groups=2, conv3x3_groups=1)
        self.deconv5 = DeconvUnit(n_cat_filter, filter_num, 25, 38, 50, 75, first_groups=2, conv3x3_groups=2)
        self.deconv6 = DeconvUnit(n_cat_filter, filter_num, 50, 75, 100, 150, first_groups=4, conv3x3_groups=2)

        '''
        if typ == 'conv':
            self.deconv1 = DeconvUnit2(filter_num, filter_num, (4, 3), 1)
            self.deconv2 = DeconvUnit2(n_cat_filter, filter_num, (3, 4), 1)
            self.deconv3 = DeconvUnit2(n_cat_filter, filter_num, (4, 4), 1)
            self.deconv4 = DeconvUnit2(n_cat_filter, filter_num, (3, 4), 1)
            self.deconv5 = DeconvUnit2(n_cat_filter, filter_num, (4, 3), 1)
            self.deconv6 = DeconvUnit2(n_cat_filter, filter_num, (3, 3), 1)
        elif typ == 'pool':
            self.deconv1 = DeconvUnitPool(filter_num, filter_num, (4, 3), 1)
            self.deconv2 = DeconvUnitPool(n_cat_filter, filter_num, (3, 4), 1)
            self.deconv3 = DeconvUnitPool(n_cat_filter, filter_num, (4, 4), 1)
            self.deconv4 = DeconvUnitPool(n_cat_filter, filter_num, (3, 4), 1)
            self.deconv5 = DeconvUnitPool(n_cat_filter, filter_num, (4, 3), 1)
            self.deconv6 = DeconvUnitPool(n_cat_filter, filter_num, (3, 3), 1)
        elif typ == 'crop':
            self.deconv1 = DeconvUnitCrop(filter_num, filter_num, (5, 4))
            self.deconv2 = DeconvUnitCrop(n_cat_filter, filter_num, (10, 7))
            self.deconv3 = DeconvUnitCrop(n_cat_filter, filter_num, (19, 13))
            self.deconv4 = DeconvUnitCrop(n_cat_filter, filter_num, (38, 25))
            self.deconv5 = DeconvUnitCrop(n_cat_filter, filter_num, (75, 50))
            self.deconv6 = DeconvUnitCrop(n_cat_filter, filter_num, (150, 100))

        # fusion
        self.conv_fusion6 = conv1x1(filter_num, filter_num, bn=self.bn)
        self.seg_score = conv1x1(filter_num, 1, bn=False, act=False)
        #self.seg_score = conv1x1(filter_num, 1, bn=self.bn, act=False)

    def forward(self, x):
        x = self.conv1(x)
        res_conv2 = self.conv2(x)
        res_conv3 = self.conv3(res_conv2)
        res_conv4 = self.conv4(res_conv3)
        res_conv5 = self.conv5(res_conv4)
        res_conv6 = self.conv6(res_conv5)

        x = self.conv7(res_conv6)
        x = self.deconv1(x)

        x = torch.cat((res_conv6, x), 1)
        res_deconv2 = self.deconv2(x)
        x = torch.cat((res_conv5, res_deconv2), 1)

        res_deconv3 = self.deconv3(x)
        x = torch.cat((res_conv4, res_deconv3), 1)

        res_deconv4 = self.deconv4(x)
        x = torch.cat((res_conv3, res_deconv4), 1)

        res_deconv5 = self.deconv5(x)
        x = torch.cat((res_conv2, res_deconv5), 1)
        x = self.deconv6(x)

        x = self.conv_fusion6(x)
        x = F.sigmoid(self.seg_score(x))
        return x


class SegShuffleHalf(nn.Module):
    def __init__(self, input_chs=3, filter_num=64, typ='conv'):
        super(SegShuffleHalf, self).__init__()
        self.bn = True
        # nn.PixelShuffle
        # encoder
        # self.conv1 = conv3x3(input_chs, filter_num, stride=1, padding=1, group=1, bn=self.bn)
        self.conv2 = conv3x3(input_chs, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv3 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv4 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv5 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv6 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)

        self.conv7 = conv3x3(filter_num, filter_num, stride=1, padding=0, group=1, bn=self.bn)

        # decoder
        n_cat_filter = filter_num * 2

        self.deconv1 = DeconvUnit2(filter_num, filter_num, (4, 3), 1)
        self.deconv2 = DeconvUnit2(n_cat_filter, filter_num, (3, 4), 1)
        self.deconv3 = DeconvUnit2(n_cat_filter, filter_num, (4, 4), 1)
        self.deconv4 = DeconvUnit2(n_cat_filter, filter_num, (3, 4), 1)
        self.deconv5 = DeconvUnit2(n_cat_filter, filter_num, (4, 3), 1)
        self.deconv6 = DeconvUnit2(n_cat_filter, filter_num, (3, 3), 1)

        # fusion
        # self.conv_fusion6 = conv1x1(filter_num, filter_num, bn=self.bn)
        self.seg_score = conv1x1(filter_num, 1, bn=False, act=False)
        # self.seg_score = conv1x1(filter_num, 1, bn=self.bn, act=False)

    def forward(self, x):
        # x = self.conv1(x)
        res_conv2 = self.conv2(x)
        res_conv3 = self.conv3(res_conv2)
        res_conv4 = self.conv4(res_conv3)
        res_conv5 = self.conv5(res_conv4)
        res_conv6 = self.conv6(res_conv5)

        x = self.conv7(res_conv6)
        x = self.deconv1(x)

        x = torch.cat((res_conv6, x), 1)
        res_deconv2 = self.deconv2(x)
        x = torch.cat((res_conv5, res_deconv2), 1)

        res_deconv3 = self.deconv3(x)
        x = torch.cat((res_conv4, res_deconv3), 1)

        res_deconv4 = self.deconv4(x)
        x = torch.cat((res_conv3, res_deconv4), 1)

        res_deconv5 = self.deconv5(x)
        x = torch.cat((res_conv2, res_deconv5), 1)
        x = self.deconv6(x)

        # x = self.conv_fusion6(x)
        x = F.sigmoid(self.seg_score(x))
        return x


class DeconvUnit3(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, padding, gf=1, upscale_factor=2):
        # type: (object, object, object, object, object, object) -> object
        super(DeconvUnit3, self).__init__()

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        conv_in_chs = in_chs // (upscale_factor * upscale_factor)
        layers = [nn.Conv2d(conv_in_chs, out_chs, kernel_size=kernel_size, padding=padding, groups=conv_in_chs // gf),
                  nn.BatchNorm2d(out_chs),
                  nn.ReLU(inplace=True)
                  ]
        self.conv_bn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.conv_bn(x)
        return x


class SegShuffleMob(nn.Module):
    def __init__(self, input_chs=3, filter_num=64):
        super(SegShuffleMob, self).__init__()
        self.bn = True
        # nn.PixelShuffle
        # encoder
        # self.conv1 = conv3x3(input_chs, filter_num, stride=1, padding=1, group=1, bn=self.bn)
        self.conv2 = conv3x3(3, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv3 = conv_dw(filter_num, filter_num, stride=2, pad=1, bn=self.bn)
        self.conv4 = conv_dw(filter_num, filter_num, stride=2, pad=1, bn=self.bn)
        self.conv5 = conv_dw(filter_num, filter_num, stride=2, pad=1, bn=self.bn)
        self.conv6 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv7 = conv3x3(filter_num, filter_num, stride=1, padding=0, group=1, bn=self.bn)

        # decoder
        n_cat_filter = filter_num * 2

        self.deconv1 = DeconvUnit2(filter_num, filter_num, (4, 3), 1)
        self.deconv2 = DeconvUnit3(n_cat_filter, filter_num, (3, 4), 1)
        self.deconv3 = DeconvUnit3(n_cat_filter, filter_num, (4, 4), 1)
        self.deconv4 = DeconvUnit3(n_cat_filter, filter_num, (3, 4), 1)
        self.deconv5 = DeconvUnit3(n_cat_filter, filter_num, (4, 3), 1)
        self.deconv6 = DeconvUnit2(n_cat_filter, filter_num, (3, 3), 1)

        # fusion
        self.conv_fusion6 = conv1x1(filter_num, filter_num, bn=self.bn)
        self.seg_score = conv1x1(filter_num, 1, bn=False, act=False)
        # self.seg_score = conv1x1(filter_num, 1, bn=self.bn, act=False)

    def forward(self, x):
        # x = self.conv1(x)
        res_conv2 = self.conv2(x)
        res_conv3 = self.conv3(res_conv2)
        res_conv4 = self.conv4(res_conv3)
        res_conv5 = self.conv5(res_conv4)
        res_conv6 = self.conv6(res_conv5)

        x = self.conv7(res_conv6)
        x = self.deconv1(x)

        x = torch.cat((res_conv6, x), 1)
        res_deconv2 = self.deconv2(x)
        x = torch.cat((res_conv5, res_deconv2), 1)

        res_deconv3 = self.deconv3(x)
        x = torch.cat((res_conv4, res_deconv3), 1)

        res_deconv4 = self.deconv4(x)
        x = torch.cat((res_conv3, res_deconv4), 1)

        res_deconv5 = self.deconv5(x)
        x = torch.cat((res_conv2, res_deconv5), 1)
        x = self.deconv6(x)

        x = self.conv_fusion6(x)
        x = F.sigmoid(self.seg_score(x))
        return x


class SegShuffleAdd(nn.Module):
    def __init__(self, input_chs=3, filter_num=64, typ='conv'):
        super(SegShuffleAdd, self).__init__()
        self.bn = True
        # nn.PixelShuffle
        # encoder
        self.conv1 = conv3x3(input_chs, filter_num, stride=1, padding=1, group=1, bn=self.bn)
        self.conv2 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv3 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv4 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv5 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv6 = conv3x3(filter_num, filter_num, stride=2, padding=1, group=1, bn=self.bn)
        self.conv7 = conv3x3(filter_num, filter_num, stride=1, padding=0, group=1, bn=self.bn)

        # decoder
        n_cat_filter = filter_num * 2
        '''
        self.deconv1 = DeconvUnit(filter_num, filter_num, 2, 3, 4, 5, first_groups=4, conv3x3_groups=2)
        self.deconv2 = DeconvUnit(n_cat_filter, filter_num, 4, 5, 7, 10, first_groups=2, conv3x3_groups=4)
        self.deconv3 = DeconvUnit(n_cat_filter, filter_num, 7, 10, 13, 19, first_groups=1, conv3x3_groups=2)
        self.deconv4 = DeconvUnit(n_cat_filter, filter_num, 13, 19, 25, 38, first_groups=2, conv3x3_groups=1)
        self.deconv5 = DeconvUnit(n_cat_filter, filter_num, 25, 38, 50, 75, first_groups=2, conv3x3_groups=2)
        self.deconv6 = DeconvUnit(n_cat_filter, filter_num, 50, 75, 100, 150, first_groups=4, conv3x3_groups=2)

        '''
        if typ == 'conv':
            self.deconv1 = DeconvUnit2(filter_num, filter_num, (4, 3), 1)
            self.deconv2 = DeconvUnit2(filter_num, filter_num, (3, 4), 1)
            self.deconv3 = DeconvUnit2(filter_num, filter_num, (4, 4), 1)
            self.deconv4 = DeconvUnit2(filter_num, filter_num, (3, 4), 1)
            self.deconv5 = DeconvUnit2(filter_num, filter_num, (4, 3), 1)
            self.deconv6 = DeconvUnit2(filter_num, filter_num, (3, 3), 1)
        elif typ == 'pool':
            self.deconv1 = DeconvUnitPool(filter_num, filter_num, (4, 3), 1)
            self.deconv2 = DeconvUnitPool(filter_num, filter_num, (3, 4), 1)
            self.deconv3 = DeconvUnitPool(filter_num, filter_num, (4, 4), 1)
            self.deconv4 = DeconvUnitPool(filter_num, filter_num, (3, 4), 1)
            self.deconv5 = DeconvUnitPool(filter_num, filter_num, (4, 3), 1)
            self.deconv6 = DeconvUnitPool(filter_num, filter_num, (3, 3), 1)
        elif typ == 'crop':
            self.deconv1 = DeconvUnitCrop(filter_num, filter_num, (5, 4))
            self.deconv2 = DeconvUnitCrop(filter_num, filter_num, (10, 7))
            self.deconv3 = DeconvUnitCrop(filter_num, filter_num, (19, 13))
            self.deconv4 = DeconvUnitCrop(filter_num, filter_num, (38, 25))
            self.deconv5 = DeconvUnitCrop(filter_num, filter_num, (75, 50))
            self.deconv6 = DeconvUnitCrop(filter_num, filter_num, (150, 100))

        # fusion
        self.conv_fusion6 = conv1x1(filter_num, filter_num, bn=self.bn)
        self.seg_score = conv1x1(filter_num, 1, bn=False, act=False)
        # self.seg_score = conv1x1(filter_num, 1, bn=self.bn, act=False)

    def forward(self, x):
        x = self.conv1(x)
        res_conv2 = self.conv2(x)
        res_conv3 = self.conv3(res_conv2)
        res_conv4 = self.conv4(res_conv3)
        res_conv5 = self.conv5(res_conv4)
        res_conv6 = self.conv6(res_conv5)

        x = self.conv7(res_conv6)
        x = self.deconv1(x)

        x = res_conv6 + x
        res_deconv2 = self.deconv2(x)
        x = res_conv5 + res_deconv2

        res_deconv3 = self.deconv3(x)
        x = res_conv4 + res_deconv3

        res_deconv4 = self.deconv4(x)
        x = res_conv3 + res_deconv4

        res_deconv5 = self.deconv5(x)
        x = res_conv2 + res_deconv5
        x = self.deconv6(x)

        x = self.conv_fusion6(x)
        x = F.sigmoid(self.seg_score(x))
        return x


def speed_cpu(model, name):
    input = torch.rand(1, 3, 150, 100)
    input = Variable(input, volatile=True)
    t1 = time.clock()
    for _ in range(100):
        model(input)
    t2 = time.clock()
    print('%10s : %f' % (name, (t2 - t1) / 100.0))


def speed_gpu(model, name, gpu_id, batch_size):
    input = torch.rand(batch_size, 3, 150, 100).cuda(gpu_id)
    input = Variable(input, volatile=True)
    model(input)
    t1 = time.clock()
    input = torch.rand(batch_size, 3, 150, 100).cuda(gpu_id)
    input = Variable(input, volatile=True)
    for _ in range(100):
        model(input)
    t2 = time.clock()
    print('%10s : %f' % (name, (t2 - t1) / 100.0))


def test_gpu(gpu_id=1, batch_size=32):
    import torch.backends.cudnn as cudnn
    torch.cuda.device(gpu_id)
    cudnn.benchmark = True

    segNet = SegNet(filter_num=64).cuda(gpu_id)
    segMobNet = SegMobileNet(filter_num=64).cuda(gpu_id)
    segUp = SegUpsample(filter_num=64).cuda(gpu_id)
    segShuffle = SegShuffle(filter_num=64).cuda(gpu_id)
    segShuffle_pool = SegShuffle(filter_num=64, typ='pool').cuda(gpu_id)
    segShuffle_crop = SegShuffle(filter_num=64, typ='crop').cuda(gpu_id)
    segShuffle_add = SegShuffleAdd(filter_num=64, typ='conv').cuda(gpu_id)

    speed_gpu(segMobNet, 'SegMobileNet', gpu_id, batch_size)
    speed_gpu(segUp, 'SegUpsample', gpu_id, batch_size)
    speed_gpu(segShuffle, 'SegShuffle', gpu_id, batch_size)
    speed_gpu(segShuffle_pool, 'SegShufflePool', gpu_id, batch_size)
    speed_gpu(segShuffle_crop, 'SegShuffleCrop', gpu_id, batch_size)
    speed_gpu(segShuffle_add, 'SegShuffleAdd', gpu_id, batch_size)
    speed_gpu(segNet, 'SegNet', gpu_id, batch_size)


def test_cpu(f_num=64):
    segNet = SegNet(filter_num=f_num)
    segMobNet = SegMobileNet(filter_num=f_num)
    segUp = SegUpsample(filter_num=f_num)
    segShuffle = SegShuffle(filter_num=f_num)
    segShuffle_pool = SegShuffle(filter_num=f_num, typ='pool')
    segShuffle_crop = SegShuffle(filter_num=f_num, typ='crop')
    segShuffle_add = SegShuffleAdd(filter_num=f_num, typ='conv')

    speed_cpu(segMobNet, 'SegMobileNet')
    speed_cpu(segUp, 'SegUpsample')
    speed_cpu(segShuffle, 'SegShuffle')
    speed_cpu(segShuffle_pool, 'SegShufflePool')
    speed_cpu(segShuffle_crop, 'SegShuffleCrop')
    speed_cpu(segShuffle_add, 'SegShuffleAdd')
    speed_cpu(segNet, 'SegNet')


if __name__ == '__main__':
    test_gpu(3, 64)
    test_cpu(64)
