import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.densenet import densenet121


class TargetDensenetBase(nn.Module):

    def __init__(self, arch='densenet121'):
        super(TargetDensenetBase, self).__init__()

        self.f = []

        if arch == 'densenet121':
            model_name = densenet121()
        else:
            raise NotImplementedError

        if 'densenet' in arch:
            for name, module in model_name.named_children():
                if name == 'conv1':
                    module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                    self.f.append(module)

            self.f = nn.Sequential(*self.f)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)

        return feature


class TargetDensenet(nn.Module):

    def __init__(self, feature_dim=128, arch='densenet121'):
        super(TargetDensenet, self).__init__()

        self.f = TargetDensenetBase(arch)
        if arch == 'densenet121':
            projection_model = nn.Sequential(nn.Linear(1024, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        else:
            raise NotImplementedError

        self.g = projection_model
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(512)


    def forward(self, x):

        feature = self.f(x)

        return self.adaptiveavgpool(feature)

