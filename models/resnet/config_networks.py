from torch import nn
from .resnet import resnet18


class ConfigureNetworks:
    """ Creates the feature extractor networks.
    """
    def __init__(
            self,
            feature_size,
            hyperpixel_ids,
            pretrained_resnet_path = None,
            freeze_feature_extractor = False,
            first_padding=3,
            hyperpixel_pooling='bilinear'
            ):

        self.feature_extractor = resnet18(
                feature_size,
                hyperpixel_ids,
                pretrained=True,
                pretrained_model_path=pretrained_resnet_path,
                first_padding=first_padding,
                hyperpixel_pooling=hyperpixel_pooling
            )

        if freeze_feature_extractor:
            for name, param in self.feature_extractor.named_parameters():
                param.requires_grad = False
                if 'layer4' in name and 'conv' in name:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')

    def get_feature_extractor(self):
        return self.feature_extractor