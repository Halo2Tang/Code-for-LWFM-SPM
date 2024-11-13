"""
This code was based on the file resnet.py (https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
from the pytorch/vision library (https://github.com/pytorch/vision).

The original license is included below:

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch.nn as nn
import torchvision.models as models
from .hyperpixel_pooling import AdaptiveBilinearInterpolatePooling, AdaptiveAveragePooling

__all__ = ['ResNet', 'resnet18']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class InitialBlock(nn.Module):
    def __init__(self, inplanes, first_padding, bn_fn) -> None:
        super(InitialBlock, self).__init__()
        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=first_padding, bias=False)
        self.bn1 = bn_fn(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3,2,1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, bn_fn, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = bn_fn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = bn_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block,
            layers,
            channels,
            feature_size,
            hyperpixel_ids,
            first_padding=3,
            hyperpixel_pooling: str = 'bilinear'):
        super(ResNet, self).__init__()
        self.bn_fn = nn.BatchNorm2d
        self.layers = layers
        self.channels = channels
        self.feature_size = feature_size
        self.hyperpixel_ids = hyperpixel_ids
        self.hyperpixel_dims = [self.channels[i] for i in self.hyperpixel_ids]
        self.encoder_dim = sum(self.hyperpixel_dims)

        inplanes = self.inplanes = 64
        self.initial_block = InitialBlock(inplanes, first_padding, self.bn_fn)
        self.layer1 = self._make_layer(block, inplanes, layers[0], self.bn_fn)
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], self.bn_fn, stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], self.bn_fn, stride=2)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], self.bn_fn, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.hyperpixel_poolings = nn.ModuleList([
            get_hyperpixel_pooling(hyperpixel_pooling)(
                feature_size=self.feature_size) for _ in self.hyperpixel_ids
        ])

    @property
    def output_size(self):
        return self.encoder_dim

    def _make_layer(self, block, planes, blocks, bn_fn, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                bn_fn(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, bn_fn, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_fn))
        return nn.Sequential(*layers)

    def forward(self, x):
        hyperpixel_id = 0
        feats = []
        x = self.initial_block(x)
        if hyperpixel_id in self.hyperpixel_ids:
            feats.append(x.clone())
        hyperpixel_id += 1

        for lid in range(len(self.layers)):
            for block in range(len(self.__getattr__('layer%d' % (lid + 1)))):
                x = self.__getattr__('layer%d' % (lid + 1))[block](x)
                if hyperpixel_id in self.hyperpixel_ids:
                    feats.append(x.clone())
                hyperpixel_id += 1


        for idx, feat in enumerate(feats):
            feats[idx] = self.hyperpixel_poolings[idx](feat)
        return feats


def get_hyperpixel_pooling(hyperpixel_pooling: str = 'bilinear'):
    if hyperpixel_pooling == 'bilinear':
        return AdaptiveBilinearInterpolatePooling
    elif hyperpixel_pooling == 'average':
        return AdaptiveAveragePooling

def resnet18(
        feature_size,
        hyperpixel_ids,
        pretrained=False,
        pretrained_model_path=None,
        first_padding=3,
        hyperpixel_pooling='bilinear',
        **kwargs):
    """
        Constructs a ResNet-18 model.
    """
    channels = [64] + [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2
    model = ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        channels,
        feature_size,
        hyperpixel_ids,
        first_padding,
        hyperpixel_pooling,
        **kwargs)

    if pretrained:
        state_dict = models.resnet18(pretrained=True).state_dict()
        update_dict = {}
        for key in list(state_dict.keys())[:6]:
            data = state_dict.pop(key)
            update_dict[f'initial_block.{key}'] = data
        state_dict.update(update_dict)
        model.load_state_dict(
            state_dict, strict=False)
    return model

def resnet34(
        feature_size,
        hyperpixel_ids,
        pretrained=False,
        pretrained_model_path=None,
        first_padding=3,
        hyperpixel_pooling='bilinear',
        **kwargs):
    """
        Constructs a ResNet-18 model.
    """
    channels = [64] + [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3
    model = ResNet(
        BasicBlock,
        [3, 4, 6, 3],
        channels,
        feature_size,
        hyperpixel_ids,
        first_padding,
        hyperpixel_pooling,
        **kwargs)

    if pretrained:
        state_dict = models.resnet34(pretrained=True).state_dict()
        update_dict = {}
        for key in list(state_dict.keys())[:6]:
            data = state_dict.pop(key)
            update_dict[f'initial_block.{key}'] = data
        state_dict.update(update_dict)
        model.load_state_dict(
            state_dict, strict=False)
    return model

