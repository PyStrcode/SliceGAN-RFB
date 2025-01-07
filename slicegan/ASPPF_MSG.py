import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import PIL.Image as Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.in_planes = in_planes
        if bn:
            self.conv = spectral_norm(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=False))
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            self.relu = nn.ReLU(inplace=True) if relu else None
        else:
            self.conv = spectral_norm(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=False))
            self.bn = None
            self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class BasicRFB(nn.Module):
    """

    """
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 1

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups,bn=False, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), bn=False, groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 1,
                      dilation=vision + 1, bn=False, relu=False, groups=groups)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, bn=False, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), bn=False, groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2,
                      dilation=vision + 2, bn=False, relu=False, groups=groups)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, bn=False, relu=False),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1, bn=False, groups=groups),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=1, padding=1, bn=False,groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4,
                      dilation=vision + 4, bn=False, relu=False, groups=groups)
        )

        self.ConvLinear = BasicConv(2 * inter_planes, out_planes, kernel_size=1, stride=1,bn=False, relu=True)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bn=False, relu=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        short = self.shortcut(x)
        x_list = [x0, x1, x2]

        out = []
        for i in range(3):
            x_i = self.ConvLinear(x_list[i])
            x_i = x_i * self.scale + short
            out.append(self.relu(x_i))
        return out

