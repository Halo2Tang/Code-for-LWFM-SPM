import torch.nn as nn
import torchvision.models as models
from .hyperpixel_pooling import AdaptiveBilinearInterpolatePooling, AdaptiveAveragePooling

__all__ = ['ResNet']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class InitialBlock(nn.Module):
    def __init__(self, inplanes, first_padding, bn_fn) -> None:
        super(InitialBlock, self).__init__()
        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=first_padding, bias=False)
        self.bn1 = bn_fn(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, bn_fn, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = bn_fn(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = bn_fn(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = bn_fn(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

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

def resnet50(
        feature_size,
        hyperpixel_ids,
        pretrained=False,
        pretrained_model_path=None,
        first_padding=3,
        hyperpixel_pooling='bilinear',
        **kwargs):
    """
        Constructs a ResNet-50 model.
    """
    channels = [64] + [256] * 3 + [512] * 4 + [1024] * 6 + [2048] * 3
    model = ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        channels,
        feature_size,
        hyperpixel_ids,
        first_padding,
        hyperpixel_pooling,
        **kwargs)

    if pretrained:
        state_dict = models.resnet50(pretrained=True).state_dict()
        update_dict = {}
        for key in list(state_dict.keys())[:6]:
            data = state_dict.pop(key)
            update_dict[f'initial_block.{key}'] = data
        state_dict.update(update_dict)
        model.load_state_dict(
            state_dict, strict=False)
    return model
