from torchvision import transforms
from .dataset_base import DATAPAIR, TESTDATA
import numpy as np

test_transform_cifar10 = transforms.Compose([
    # transforms.Resize(32),  # if the model is vgg
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_stl10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

test_transform_CLIP = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])

test_transform_imagenet = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])


fmnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def get_downstream_fmnist(args):
    training_file_name = 'train.npz'
    testing_file_name = 'test.npz'

    if args.pretraining_dataset == 'imagenet':
        test_transform = test_transform_imagenet
    elif args.pretraining_dataset == 'clip':
        test_transform = test_transform_CLIP
    elif args.pretraining_dataset == 'cifar10':
        test_transform = test_transform_cifar10
    elif args.pretraining_dataset == 'stl10':
        test_transform = test_transform_stl10

    downstream_training_data = TESTDATA(numpy_file=args.data_dir+training_file_name, class_type=fmnist_classes, transform=test_transform)
    testing_data = TESTDATA(numpy_file=args.data_dir+testing_file_name, class_type=fmnist_classes, transform=test_transform)

    return downstream_training_data, testing_data

