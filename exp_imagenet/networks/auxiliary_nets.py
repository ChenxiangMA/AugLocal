import torch
import math
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


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


class AuxClassifier(nn.Module):
    def __init__(self, inplanes, net_config='1c2f', widen=1, layer_index=0,
                 aux_net_depth=3, net_dim=None, is_bottleneck=False):
        super(AuxClassifier, self).__init__()

        assert inplanes in [64, 128, 256, 512]

        # assert net_config in ['0c1f', '0c2f', '0c3f', '1c1f', '1c2f', '2c2f', '3c1f', '4c1f', '3c1ff']
        # net_config = '1c1f'
        self.loss_mode = 'cross_entropy'

        self.criterion = nn.CrossEntropyLoss()
        self.fc_out_channels = 1000
        if is_bottleneck:
            self.block = Bottleneck
        else:
            self.block = BasicBlock
        if net_config == 'unifSamp':
            # net_dim = [64] * 4 + [128] * 4 + [256] * 6  + [512] * 3    #  resnet34
            # net_dim = [16] * 19 + [32] * 18 + [64] * 18 # resnet110
            n_layers = len(net_dim)
            if aux_net_depth >= n_layers - 1 - layer_index:
                aux_layer_index = [i for i in range(layer_index + 1, n_layers)]
            else:
                aux_layer_index = [round(layer_index + (n_layers - layer_index - 1) / aux_net_depth * i) for i in
                                   range(1, aux_net_depth + 1)]
            aux_net = []
            num_pool = 0
            if layer_index == 0:
                aux_inplanes = inplanes
            else:
                aux_inplanes = inplanes * self.block.expansion
            for index_i in aux_layer_index:
                assert net_dim[index_i] >= inplanes
                num_pool = num_pool + int(torch.log2(torch.tensor(int(net_dim[index_i] // inplanes))))
                if num_pool >= 1:
                    stride = 2
                    num_pool -= 1
                else:
                    stride = 1
                # if inplanes != net_dim[index_i]:
                #     stride = 2
                # else:
                #     stride = 1
                aux_net.append(self._make_layer(self.block, aux_inplanes, int(net_dim[index_i] * widen), stride=stride))
                aux_inplanes = int(net_dim[index_i] * self.block.expansion * widen)
                inplanes = net_dim[index_i]
            aux_net.extend([nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten(),
                            # nn.BatchNorm1d(hidden_dim),
                            nn.Linear(512 * self.block.expansion, self.fc_out_channels)])
            self.head = nn.Sequential(*aux_net)
        else:
            raise NotImplementedError

    def _make_layer(self, block, inplanes, planes, stride=1):
        norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        return block(inplanes, planes, stride, downsample)

    def forward(self, x, target):
        features = self.head(x)
        loss = self.criterion(features, target)

        return loss, features.detach()


class AuxClassifier_VGG(nn.Module):
    def __init__(self, inplanes, net_config='unifSamp', widen=1, layer_index=0, aux_net_depth=3,
                 net_dim=None):
        super(AuxClassifier_VGG, self).__init__()

        self.criterion = nn.CrossEntropyLoss()
        self.fc_out_channels = 1000

        if net_config == 'unifSamp':
            # net_dim = [64] * 2 + [128] * 2 + [256] * 4 + [512] * 4 + [512] * 4 # vgg19
            print(f"net_dim: {net_dim}")
            n_layers = len(net_dim)
            if aux_net_depth >= n_layers - 1 - layer_index:
                aux_layer_index = [i for i in range(layer_index + 1, n_layers)]
            else:
                aux_layer_index = [round(layer_index + (n_layers - layer_index - 1) / aux_net_depth * i) for i in
                                   range(1, aux_net_depth + 1)]
            aux_net = []
            for index_i in aux_layer_index:
                if inplanes != net_dim[index_i]:
                    aux_net.append(nn.MaxPool2d(kernel_size=2, stride=2))
                aux_net.extend([nn.Conv2d(inplanes, net_dim[index_i], kernel_size=3, padding=1),
                                nn.BatchNorm2d(net_dim[index_i]),
                                nn.ReLU(inplace=True)])
                inplanes = net_dim[index_i]
            aux_net.extend([  # nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.AdaptiveAvgPool1d(512),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(True),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(True),
                nn.Linear(512, self.fc_out_channels),
            ])
            self.head = nn.Sequential(*aux_net)
        if net_config == 'DGL':  # ref: https://github.com/eugenium/DGL/blob/master/auxiliary_nets_study/models.py
            if inplanes == 64:
                size_dgl = 224  # cifar10 / svhn
                # size_dgl = 96 # stl10
                mlp_in_size = min(math.ceil(size_dgl / 4), 2)
                mlp_feat = inplanes * (mlp_in_size) * (mlp_in_size)
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(math.ceil(size_dgl / 4)),
                    nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(inplanes),
                    nn.ReLU(),
                    nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(inplanes),
                    nn.ReLU(),
                    nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(inplanes),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(mlp_in_size),
                    nn.Flatten(),
                    nn.Linear(mlp_feat, mlp_feat, bias=False),
                    nn.BatchNorm1d(mlp_feat),
                    nn.ReLU(),
                    nn.Linear(mlp_feat, mlp_feat, bias=False),
                    nn.BatchNorm1d(mlp_feat),
                    nn.ReLU(),
                    nn.Linear(mlp_feat, self.fc_out_channels),
                )
            elif inplanes == 128:
                size_dgl = int(224 // 2)  # cifar10 / svhn
                # size_dgl = 48 # stl10
                mlp_in_size = min(math.ceil(size_dgl / 4), 2)
                mlp_feat = inplanes * (mlp_in_size) * (mlp_in_size)
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(math.ceil(size_dgl / 4)),
                    nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(inplanes),
                    nn.ReLU(),
                    nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(inplanes),
                    nn.ReLU(),
                    nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(inplanes),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(mlp_in_size),
                    nn.Flatten(),
                    nn.Linear(mlp_feat, mlp_feat, bias=False),
                    nn.BatchNorm1d(mlp_feat),
                    nn.ReLU(),
                    nn.Linear(mlp_feat, mlp_feat, bias=False),
                    nn.BatchNorm1d(mlp_feat),
                    nn.ReLU(),
                    nn.Linear(mlp_feat, self.fc_out_channels),
                )
            elif inplanes == 256:
                # size_dgl = 8 # cifar10 / svhn
                size_dgl = int(224 // 4)  # stl10
                mlp_in_size = min(math.ceil(size_dgl / 4), 2)
                mlp_feat = inplanes * (mlp_in_size) * (mlp_in_size)
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(math.ceil(size_dgl / 4)),
                    nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(inplanes),
                    nn.ReLU(),
                    nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(inplanes),
                    nn.ReLU(),
                    nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(inplanes),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(mlp_in_size),
                    nn.Flatten(),
                    nn.Linear(mlp_feat, mlp_feat, bias=False),
                    nn.BatchNorm1d(mlp_feat),
                    nn.ReLU(),
                    nn.Linear(mlp_feat, mlp_feat, bias=False),
                    nn.BatchNorm1d(mlp_feat),
                    nn.ReLU(),
                    nn.Linear(mlp_feat, self.fc_out_channels),
                )
            elif inplanes == 512:
                # size_dgl = 8 # cifar10 / svhn
                size_dgl = int(224 // 8)  # stl10
                mlp_in_size = min(math.ceil(size_dgl / 4), 2)
                mlp_feat = inplanes * (mlp_in_size) * (mlp_in_size)
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(math.ceil(size_dgl / 4)),
                    nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(inplanes),
                    nn.ReLU(),
                    nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(inplanes),
                    nn.ReLU(),
                    nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(inplanes),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(mlp_in_size),
                    nn.Flatten(),
                    nn.Linear(mlp_feat, mlp_feat, bias=False),
                    nn.BatchNorm1d(mlp_feat),
                    nn.ReLU(),
                    nn.Linear(mlp_feat, mlp_feat, bias=False),
                    nn.BatchNorm1d(mlp_feat),
                    nn.ReLU(),
                    nn.Linear(mlp_feat, self.fc_out_channels),
                )

    def forward(self, x, target):

        features = self.head(x)
        loss = self.criterion(features, target)

        return loss, features.detach()
