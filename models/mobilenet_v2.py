import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenetv2 import mobilenet_v2


class TargetMobilenet_v2Base(nn.Module):

    def __init__(self, arch='mobilenet_v2'):
        super(TargetMobilenet_v2Base, self).__init__()

        self.f = []

        if arch == 'mobilenet_v2':
            model_name = mobilenet_v2()
        else:
            raise NotImplementedError

        self.f = model_name.features

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)

        return feature


class TargetMobilenet_v2(nn.Module):

    def __init__(self, feature_dim=128, arch='mobilenet_v2'):
        super(TargetMobilenet_v2, self).__init__()

        self.f = TargetMobilenet_v2Base(arch)
        if arch == 'mobilenet_v2':
            projection_model = nn.Sequential(nn.Linear(1280, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        else:
            raise NotImplementedError

        self.g = projection_model
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(512)


    def forward(self, x):

        feature = self.f(x)

        return self.adaptiveavgpool(feature)

