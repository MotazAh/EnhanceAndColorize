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
  pass