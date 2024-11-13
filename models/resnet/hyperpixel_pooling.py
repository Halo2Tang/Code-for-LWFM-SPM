import torch
from torch import nn
import torch.nn.functional as F


class AdaptiveBilinearInterpolatePooling(nn.Module):
    def __init__(self, feature_size):
        super(AdaptiveBilinearInterpolatePooling, self).__init__()
        self.feature_size = feature_size

    def forward(self, x):
        return F.interpolate(x, self.feature_size, None, 'bilinear', True)


class AdaptiveAveragePooling(nn.Module):
    def __init__(self, feature_size):
        super(AdaptiveAveragePooling, self).__init__()
        self.feature_size = feature_size

    def forward(self, x):
        return F.adaptive_avg_pool2d(x, [self.feature_size, self.feature_size])