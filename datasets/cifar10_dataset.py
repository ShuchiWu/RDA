from torchvision import transforms
from .dataset_base import DATAPAIR, TESTDATA
from .watermark_dataset import TestBackdoor
import numpy as np

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_stl10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])


test_transform_imagenet = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

test_transform_CLIP = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])

cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def get_pretraining_cifar10(data_dir):

    train_data = DATAPAIR(numpy_file=data_dir + "train.npz", class_type= cifar10_classes, transform=train_transform)
    memory_data = TESTDATA(numpy_file=data_dir + "train.npz", class_type=cifar10_classes, transform=test_transform_cifar10)
    test_data  = TESTDATA(numpy_file=data_dir + "test.npz", class_type= cifar10_classes,transform=test_transform_cifar10)

    return train_data, memory_data, test_data


def get_downstream_cifar10(args):
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

    downstream_training_data = TESTDATA(numpy_file=args.data_dir+training_file_name, class_type=cifar10_classes, transform=test_transform)
    testing_data = TESTDATA(numpy_file=args.data_dir+testing_file_name, class_type=cifar10_classes, transform=test_transform)

    return downstream_training_data, testing_data

def get_backdoor_cifar10(args):
    training_file_name = 'train.npz'
    testing_file_name = 'test.npz'

    print('='*15, 'Loading triggered dataset for testing', '='*15)

    memory_data = TESTDATA(numpy_file=args.test_data_dir+training_file_name, class_type=cifar10_classes, transform=test_transform_cifar10)
    test_data_backdoor = TestBackdoor(numpy_file=args.test_data_dir + testing_file_name, trigger_file='trigger/trigger.npz', reference_label=args.target_label, transform=test_transform_cifar10)
    test_data_clean = TESTDATA(numpy_file=args.test_data_dir + testing_file_name, class_type=cifar10_classes, transform=test_transform_cifar10)

    return memory_data, test_data_backdoor, test_data_clean


