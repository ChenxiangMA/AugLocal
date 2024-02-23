import logging
from functools import partial
from typing import Any, cast, Dict, List, Optional, Union

import torch
import torch.nn as nn

from .configs import network_config
from .auxiliary_nets import AuxClassifier_VGG


__all__ = [
    "VGG",
    "vgg13",
]

cfgs = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

class VGG(nn.Module):
    def __init__(
        self,
        arch,
        local_module_num,
        num_classes=1000,
        init_weights=True,
        dropout=0.5,
        aux_net_config='DGL',
        aux_net_widen=1,
        aux_net_depth=1,
        layerwise_train=False,
        is_pyramid=False,
        paramid_coeff=0.2
    ):
        super().__init__()
        self.local_module_num = local_module_num
        self.features = make_layers(cfgs[arch])
        self.depth_of_net = len(cfgs[arch])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        self.is_pyramid = is_pyramid
        self.layerwise_train = layerwise_train
        if self.is_pyramid:
            self.aux_depth_list = [max(1, round(
                aux_net_depth * (1 - (i * paramid_coeff) / (self.local_module_num - 2)) + (i * paramid_coeff) / (
                        self.local_module_num - 2) * 1)) for i in range(self.local_module_num - 1)]
            logging.info(f"aux_depth_list: {self.aux_depth_list}")
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
        self.infopro_config = network_config[arch][local_module_num]
        self.net_dim = network_config[arch]['net_dim']
        for layer_index, layer_num in enumerate(self.infopro_config):
             exec('self.aux_classifier_' + str(layer_num) +
                 '= AuxClassifier_VGG(cfgs[arch][layer_num], net_config=aux_net_config, '
                 'widen=aux_net_widen, layer_index=layer_index, aux_net_depth=self.aux_depth_list[layer_index] if self.is_pyramid else aux_net_depth, net_dim=self.net_dim)')

    def forward_original(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x, target=None, criterion=None, stage_i=None, layer_i=None, hidden_logits_record=None):
        if not self.training:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
        elif self.layerwise_train:
            if layer_i < self.depth_of_net-2:
                if layer_i in [3, 6, 9, 12]:
                    x = self.features[layer_i-1](x)
                x = self.features[layer_i](x)
                loss_ixy, hidden_logits = eval('self.aux_classifier_' + str(layer_i))(
                    x, target)
                hidden_logits_record.append(hidden_logits)
                return x.detach(), loss_ixy, hidden_logits_record
            elif layer_i == self.depth_of_net-2:
                x = self.features[layer_i](x)
                x = self.features[layer_i+1](x)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                output_loss = criterion(x, target)
                return x, output_loss, hidden_logits_record
            raise NotImplementedError
        else:
            loss = 0.
            hidden_logits_record = []
            local_module_i = 0

            for layer_i in range(self.depth_of_net):
                x = self.features[layer_i](x)
                if local_module_i <= self.local_module_num - 2 and self.infopro_config[local_module_i] == layer_i:
                    loss_ixy, hidden_logits = eval('self.aux_classifier_' + str(layer_i))(
                        x, target)
                    hidden_logits_record.append(hidden_logits)
                    if self.training:
                        loss = loss + loss_ixy

                    x = x.detach()
                    local_module_i += 1
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            output_loss = criterion(x, target)
            loss = loss + output_loss
            return x, loss, hidden_logits_record, output_loss.item()


def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            layers += [BasicLayer(in_channels, v)]
            in_channels = v
    return nn.Sequential(*layers)

class BasicLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(BasicLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out




def _vgg(cfg, **kwargs):
    model = VGG(cfg, **kwargs)
    return model




def vgg13(**kwargs):
    return _vgg("vgg13", **kwargs)

