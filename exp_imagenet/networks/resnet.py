import logging

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from .configs import network_config
from .auxiliary_nets import AuxClassifier

__all__ = ['ResNet', 'resnet34', 'resnet101',
           ]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}






def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
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
    def __init__(self, block, layers, arch, local_module_num, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None,
                 aux_net_config='unifSamp',
                 aux_net_widen=1,
                 aux_net_depth=1,
                 layerwise_train=False,
                 is_pyramid=False,
                 paramid_coeff=0.2):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        wide_list = [64, 64, 128, 256, 512]
        self.inplanes = wide_list[0]
        self.groups = groups
        self.layers = layers
        self.base_width = width_per_group
        self.local_module_num = local_module_num
        self.layerwise_train = layerwise_train
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if arch == 'resnet50' or arch == 'resnet101':
            is_bottleneck = True
        else:
            is_bottleneck = False
        self.layer1 = self._make_layer(block, wide_list[1], layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, wide_list[2], layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, wide_list[3], layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, wide_list[4], layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.net_config = network_config[arch][local_module_num]
        self.net_dim = network_config[arch]['net_dim']
        self.is_pyramid = is_pyramid
        if self.is_pyramid:
            self.aux_depth_list = [max(1, round(
                aux_net_depth * (1 - (i * paramid_coeff) / (self.local_module_num - 2)) + (i * paramid_coeff) / (
                        self.local_module_num - 2) * 1)) for i in range(self.local_module_num - 1)]
            logging.info(f"aux_depth_list: {self.aux_depth_list}")

        for item in self.net_config:
            module_index, layer_index, layer_num = item

            exec('self.aux_classifier_' + str(module_index) + '_' + str(layer_index) +
                 '= AuxClassifier(wide_list[module_index], net_config=aux_net_config, '
                 'widen=aux_net_widen, layer_index=layer_num, '
                 'aux_net_depth=self.aux_depth_list[layer_num] if self.is_pyramid else aux_net_depth, '
                 'net_dim=self.net_dim, is_bottleneck=is_bottleneck)')

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, target=None, criterion=None, stage_i=None, layer_i=None, hidden_logits_record=None):
        if not self.training:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            output = self.avgpool(x)
            output = output.view(x.size(0), -1)
            output = self.fc(output)

            return output
        elif self.layerwise_train:
            if stage_i == 0:
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                loss_ixy, hidden_logits = eval('self.aux_classifier_' + str(stage_i) + '_' + str(layer_i))(
                    x, target)
                hidden_logits_record.append(hidden_logits)
                if self.training:
                    loss = loss_ixy
                return x.detach(), loss, hidden_logits_record

            elif stage_i == 4 and layer_i + 1 == self.layers[stage_i - 1]:
                x = eval('self.layer' + str(stage_i))[layer_i](x)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                loss = criterion(x, target)
                return x, loss, hidden_logits_record
            else:
                x = eval('self.layer' + str(stage_i))[layer_i](x)
                loss_ixy, hidden_logits = eval('self.aux_classifier_' + str(stage_i) + '_' + str(layer_i))(
                    x, target)
                hidden_logits_record.append(hidden_logits)
                if self.training:
                    loss = loss_ixy
                return x.detach(), loss, hidden_logits_record
        else:
            loss = 0.
            hidden_logits_record = []

            stage_i = 0
            layer_i = 0
            local_module_i = 0

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            if local_module_i <= self.local_module_num - 2:
                if self.net_config[local_module_i][0] == stage_i and self.net_config[local_module_i][1] == layer_i:
                    loss_ixy, hidden_logits = eval('self.aux_classifier_' + str(stage_i) + '_' + str(layer_i))(
                        x, target)
                    hidden_logits_record.append(hidden_logits)
                    if self.training:
                        loss = loss + loss_ixy
                    x = x.detach()
                    local_module_i += 1

            for stage_i in (1, 2, 3, 4):
                for layer_i in range(self.layers[stage_i - 1]):
                    x = eval('self.layer' + str(stage_i))[layer_i](x)

                    if local_module_i <= self.local_module_num - 2:
                        if self.net_config[local_module_i][0] == stage_i and self.net_config[local_module_i][1] == layer_i:
                            loss_ixy, hidden_logits = eval('self.aux_classifier_' + str(stage_i) + '_' + str(layer_i))(
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



def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], arch='resnet34', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model



def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], arch='resnet101',  **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model




