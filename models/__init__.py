from .simclr_model import SimCLR
from .resnet import ResNet18, ResNet34
from .imagenet_model import ImageNetResNet, TargerImageNetResNet
from .clip_model import TargetCLIP, CLIP
from .densenet import TargetDensenet
from .mobilenet_v2 import TargetMobilenet_v2


def get_encoder_architecture(args):
    if args.pretraining_dataset == 'cifar10':
        return SimCLR()
    elif args.pretraining_dataset == 'stl10':
        return SimCLR()
    elif args.pretraining_dataset == 'imagenet' and args.task == 'steal':
        return TargerImageNetResNet()
    elif args.pretraining_dataset == 'clip' and args.task == 'steal':
        return TargetCLIP(1024, 224, vision_layers=(3, 4, 6, 3), vision_width=64)
    elif args.pretraining_dataset == 'imagenet' and args.task == 'downstream_task':
        return ImageNetResNet()
    elif args.pretraining_dataset == 'clip' and args.task == 'downstream_task':
        return CLIP(1024, 224, vision_layers=(3, 4, 6, 3), vision_width=64)
    else:
        raise ValueError('Unknown pretraining dataset: {}'.format(args.pretraining_dataset))


def get_surrogate_model(args):
    if args.architecture == 'resnet18':
        return SimCLR(arch='resnet18').f
    elif args.architecture == 'resnet34':
        return SimCLR(arch='resnet34').f
    elif args.architecture == 'resnet50':
        return SimCLR(arch='resnet50').f
    else:
        raise ValueError('Unknown query dataset: {}'.format(args.query_dataset))


def get_downstream_model(args):
    if args.pretraining_dataset == 'cifar10':
        return SimCLR(arch='resnet18')
    elif args.pretraining_dataset == 'stl10':
        return SimCLR(arch='resnet18')
    elif args.pretraining_dataset == 'imagenet' and args.task == 'downstream_task':
        if 'our' not in args.encoder:
            return ImageNetResNet()
        else:
            return SimCLR(arch='resnet18')
    elif args.pretraining_dataset == 'clip' and args.task == 'downstream_task':
        if 'our' not in args.encoder:
            return CLIP(1024, 224, vision_layers=(3, 4, 6, 3), vision_width=64)
        else:
            return SimCLR(arch='resnet18')
