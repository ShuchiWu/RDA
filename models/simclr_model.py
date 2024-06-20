import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50
from torchvision.models.vgg import vgg19_bn
from torchvision.models.densenet import densenet121
from torchvision.models.mobilenetv2 import mobilenet_v2


class SimCLRBase(nn.Module):

    def __init__(self, arch='resnet18'):
        super(SimCLRBase, self).__init__()

        self.f = []

        if arch == 'resnet18':
            model_name = resnet18()
        elif arch == 'resnet34':
            model_name = resnet34()
        elif arch == 'resnet50':
            model_name = resnet50()
        elif arch == 'vgg19_bn':
            model_name = vgg19_bn()
        elif arch == 'densenet121':
            model_name = densenet121()
        elif arch == 'mobilenet_v2':
            model_name = mobilenet_v2()
        else:
            raise NotImplementedError

        if 'resnet' in arch or 'densenet' in arch:
            for name, module in model_name.named_children():
                if name == 'conv1':
                    module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                    self.f.append(module)
            self.f = nn.Sequential(*self.f)
        elif 'vgg' in arch:
            self.f = model_name.features
        elif 'mobilenet' in arch:
            self.f = model_name.features

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)

        return feature

class SimCLR(nn.Module):

    def __init__(self, feature_dim=128, arch='resnet18'):
        super(SimCLR, self).__init__()

        self.f = SimCLRBase(arch)
        if arch == 'resnet18':
            projection_model = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        elif arch == 'resnet34':
            projection_model = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        elif arch == 'resnet50':
            projection_model = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        elif arch == 'vgg19_bn':
            projection_model = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        elif arch == 'densenet121':
            projection_model = nn.Sequential(nn.Linear(1024, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        elif arch == 'mobilenet_v2':
            projection_model = nn.Sequential(nn.Linear(1280, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        else:
            raise NotImplementedError

        self.g = projection_model

    def forward(self, x):

        feature = self.f(x)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


if __name__ == "__main__":
    device = 'cuda'
    model = SimCLR().f.cuda()
    from torchsummary import summary
    summary(model, input_size=(3, 32, 32))
