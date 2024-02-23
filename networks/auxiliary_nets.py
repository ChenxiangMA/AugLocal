import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .losses import CELoss, Criterion, SimLoss


def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AuxNetwork(nn.Module):
    def __init__(self, inplanes, net_config='1c2f', class_num=10, widen=1, feature_dim=128, hidden_dim=1024,
                 layer_index=0, aux_net_depth=3, net_dim=None, image_size=32):
        super(AuxNetwork, self).__init__()

        assert inplanes in [16, 32, 64]
        assert image_size in [32, 96]

        self.feature_dim = feature_dim
        self.criterion = nn.CrossEntropyLoss()
        self.fc_out_channels = class_num

        if net_config == 'PredSim':
            from utils.others import get_pool_layer
            avg_pool, dim_in_decoder = get_pool_layer(inplanes, int(image_size // int(inplanes // 16)), hidden_dim)
            self.head = nn.Sequential(
                avg_pool,
                nn.Flatten(),
                nn.Linear(dim_in_decoder, self.fc_out_channels)
            )
        if net_config == 'DGL':  # ref: https://github.com/eugenium/DGL/blob/master/auxiliary_nets_study/models.py
            if inplanes == 16:
                size_dgl = image_size  # cifar10 / svhn
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
            elif inplanes == 32:
                size_dgl = int(image_size // 2)  # cifar10 / svhn
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
            elif inplanes == 64:
                # size_dgl = 8 # cifar10 / svhn
                size_dgl = int(image_size // 4)  # stl10
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
        if net_config == 'InfoPro':
            if inplanes == 16:
                self.head = nn.Sequential(
                    nn.Conv2d(16, int(32 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(32 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(32 * widen), int(128 * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(128 * widen), self.fc_out_channels)
                )
            elif inplanes == 32:
                self.head = nn.Sequential(
                    nn.Conv2d(32, int(64 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), int(128 * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(128 * widen), self.fc_out_channels)
                )
            elif inplanes == 64:
                self.head = nn.Sequential(
                    nn.Conv2d(64, int(64 * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), int(128 * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(128 * widen), self.fc_out_channels)
                )
        if net_config == 'unifSamp':
            # net_dim = [16] * 6 + [32] * 5 + [64] * 5 # resnet32
            # net_dim = [16] * 19 + [32] * 18 + [64] * 18 # resnet110
            n_layers = len(net_dim)
            if aux_net_depth >= n_layers - 1 - layer_index:
                # print(layer_index)
                # raise NotImplementedError
                aux_layer_index = [i for i in range(layer_index + 1, n_layers)]
            else:
                aux_layer_index = [round(layer_index + (n_layers - layer_index - 1) / aux_net_depth * i) for i in
                                   range(1, aux_net_depth + 1)]
            aux_net = []
            num_pool = 0
            aux_planes = inplanes
            for index_i in aux_layer_index:
                assert net_dim[index_i] >= inplanes
                num_pool = num_pool + int(torch.log2(torch.tensor(int(net_dim[index_i] // inplanes))))
                if num_pool >= 1:
                    stride = 2
                    num_pool -= 1
                else:
                    stride = 1
                aux_net.append(self._make_layer(inplanes, int(net_dim[index_i] * widen), stride=stride))
                inplanes = int(net_dim[index_i] * widen)
                aux_planes = net_dim[index_i]
            if aux_net_depth <= 1:
                aux_net.extend([nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(),
                                nn.BatchNorm1d(int(aux_planes * widen)),
                                nn.Linear(int(aux_planes * widen), self.fc_out_channels)])
            else:
                aux_net.extend([nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(),
                                nn.Linear(int(aux_planes * widen), self.fc_out_channels)])
            self.head = nn.Sequential(*aux_net)
        if net_config == '0c1f':
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(int(inplanes * widen), self.fc_out_channels),
            )

    def _make_layer(self, inplanes, planes, stride=1):
        block = BasicBlock
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        return block(inplanes, planes, stride, downsample)

    def forward(self, x, target=None):
        features = self.head(x)
        loss = self.criterion(features, target)
        return loss, features.detach()


class Decoder_InfoPro(nn.Module):
    def __init__(self, inplanes, image_size, interpolate_mode='bilinear', net_config='1c2f', widen=1, layer_index=0,
                 aux_net_depth=3, net_dim=None):
        super(Decoder_InfoPro, self).__init__()
        self.image_size = image_size
        assert interpolate_mode in ['bilinear', 'nearest']
        self.interpolate_mode = interpolate_mode
        self.bce_loss = nn.BCELoss()

        if net_config == '4c':
            self.decoder = nn.Sequential(
                nn.Conv2d(inplanes, int(12 * widen), kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(int(12 * widen)),
                nn.ReLU(),
                nn.Conv2d(int(12 * widen), int(12 * widen), kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(int(12 * widen)),
                nn.ReLU(),
                nn.Conv2d(int(12 * widen), int(12 * widen), kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(int(12 * widen)),
                nn.ReLU(),
                nn.Conv2d(int(12 * widen), 3, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid(),
            )
        elif net_config == '2c':
            self.decoder = nn.Sequential(
                nn.Conv2d(inplanes, int(12 * widen), kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(int(12 * widen)),
                nn.ReLU(),
                nn.Conv2d(int(12 * widen), 3, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid(),
            )
        else:
            raise NotImplementedError

    def forward(self, features, image_ori):
        if self.interpolate_mode == 'bilinear':
            features = F.interpolate(features, size=[self.image_size, self.image_size],
                                     mode='bilinear', align_corners=True)
        elif self.interpolate_mode == 'nearest':  # might be faster
            features = F.interpolate(features, size=[self.image_size, self.image_size],
                                     mode='nearest')
        else:
            raise NotImplementedError

        return self.bce_loss(self.decoder(features), image_ori)

    def _make_layer(self, inplanes, planes, stride=1):
        block = BasicBlock
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        return block(inplanes, planes, stride, downsample)


class Decoder_PredSim(nn.Module):
    def __init__(self, inplanes, class_num=10):
        super(Decoder_PredSim, self).__init__()
        self.sim_loss = SimLoss(classes=class_num)
        self.decoder = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, features, y):
        return self.sim_loss(self.decoder(features), y)
