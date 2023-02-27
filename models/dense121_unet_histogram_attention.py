from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from torchvision import models as torch_models
from torchvision.models import resnet34
from torchvision.models.resnet import BasicBlock, Bottleneck

from models.networks import _DenseBlock, _Transition, RDB, GaussianHistogram, AttentionExtractModule

# For colorization network
class DoubleConv(nn.Module):
  pass

# For colorization network
class Up(nn.Module):
  pass



# For colorization network
class Dense121UnetHistogramAttention(nn.Module):
  def __init__(self, args, color_pretrain=False, growth_rate=32, block_config=(6, 12, 24, 48),
                 num_init_features=64, bn_size=4):
    super(Dense121UnetHistogramAttention, self).__init__()
    self.color_pretrain = color_pretrain

    # First convolution
    self.features = nn.Sequential(OrderedDict([
        ('conv0_0', nn.Conv2d(1, num_init_features, kernel_size=7, stride=1,
                              padding=3, bias=False)),
        ('relu0', nn.ReLU(inplace=True)),
        ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
    ]))

  def forward(self, x, x_gray, ref, ref_gray, att_model):
    # shallow conv (First Convolution)
    feature0 = self.features.relu0(self.features.conv0_0(x))
    down0 = self.features.pool0(feature0) # Downsample by pooling

    # TODO: Change results to output of model
    results = {'output': torch.randn((1, 2, 256, 256))}
    return results

