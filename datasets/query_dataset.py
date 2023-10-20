import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset, DataLoader
from .dataset_base import TESTDATA
from torchvision.transforms import InterpolationMode


test_transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_stl10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

test_transform_imagenet = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()])

test_transform_CLIP = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


class CIFAR10QUERYDataset(Dataset):

    def __init__(self, numpy_file, indices, transform=None):
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']
        self.target = self.input_array['y']
        self.indices = indices
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[self.indices[index]]
        label = self.target[self.indices[index]]
        img = Image.fromarray(img)
        query_img = self.transform(img)

        return query_img, label

    def __len__(self):
        return len(self.indices)


class CIFAR10QUERYPairData(Dataset):

    def __init__(self, numpy_file, indices, transform_1=None, transform_2=None):
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']
        self.targets = self.input_array['y']
        self.indices = indices
        self.transform_1 = transform_1
        self.transform_2 = transform_2
        self.transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])

    def __getitem__(self, index):
        img = self.data[self.indices[index]]
        label = self.targets[self.indices[index]]
        img = Image.fromarray(img)
        if self.transform_1 is not None:
            im_1 = self.transform_1(img)
        else:
            im_1 = self.transform(img)
        if self.transform_2 is not None:
            im_2 = self.transform_2(img)
        else:
            im_2 = self.transform(img)
        return im_1, im_2, label

    def __len__(self):
        return len(self.indices)


class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: a solarized image.
        """

        return ImageOps.solarize(img)


class GBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# Here we set two sets of the same code to investigate the impact of different scales for proto and query.
class CIFARPROTODATAViewGenerator(object):
    def __init__(self, num_patch=5):
        self.num_patch = num_patch

    def __call__(self, x):
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        aug_transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.9, 0.9), ratio=(1, 1)), # the scale is critical
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GBlur(p=0.1),
            transforms.RandomApply([Solarization()], p=0.1),
            transforms.ToTensor(),
            normalize
        ])
        augmented_x = [aug_transform(x) for i in range(self.num_patch)]

        return augmented_x


class CIFARQUERYDATAViewGenerator(object):
    def __init__(self, num_patch=5):
        self.num_patch = num_patch

    def __call__(self, x):
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        aug_transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.75, 0.75), ratio=(1, 1)), # the scale is critical
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GBlur(p=0.1),
            transforms.RandomApply([Solarization()], p=0.1),
            transforms.ToTensor(),
            normalize
        ])
        augmented_x = [aug_transform(x) for i in range(self.num_patch)]

        return augmented_x


class ImageNetPROTODATAViewGenerator(object):
    def __init__(self, num_patch=5):
        self.num_patch = num_patch

    def __call__(self, x):
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        aug_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomResizedCrop(32, scale=(0.9, 0.9), ratio=(1, 1)), # the scale is critical
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GBlur(p=0.1),
            transforms.RandomApply([Solarization()], p=0.1),
            transforms.ToTensor(),
            normalize
        ])
        augmented_x = [aug_transform(x) for i in range(self.num_patch)]

        return augmented_x


class ImageNetQUERYDATAViewGenerator(object):
    def __init__(self, num_patch=5):
        self.num_patch = num_patch

    def __call__(self, x):
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        aug_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomResizedCrop(32, scale=(0.75, 0.75), ratio=(1, 1)), # the scale is critical
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GBlur(p=0.1),
            transforms.RandomApply([Solarization()], p=0.1),
            transforms.ToTensor(),
            normalize
        ])
        augmented_x = [aug_transform(x) for i in range(self.num_patch)]

        return augmented_x


class ImageNet224PROTODATAViewGenerator(object):
    def __init__(self, num_patch=5):
        self.num_patch = num_patch

    def __call__(self, x):
        aug_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomResizedCrop(224, scale=(0.9, 0.9), ratio=(1, 1)),  # the scale is critical
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GBlur(p=0.1),
            transforms.RandomApply([Solarization()], p=0.1),
            transforms.ToTensor()
        ])
        augmented_x = [aug_transform(x) for i in range(self.num_patch)]

        return augmented_x


class ImageNet224QUERYDATAViewGenerator(object):
    def __init__(self, num_patch=5):
        self.num_patch = num_patch

    def __call__(self, x):
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        aug_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomResizedCrop(224, scale=(0.75, 0.75), ratio=(1, 1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GBlur(p=0.1),
            transforms.RandomApply([Solarization()], p=0.1),
            transforms.ToTensor(),
            normalize
        ])
        augmented_x = [aug_transform(x) for i in range(self.num_patch)]

        return augmented_x


class CLIP224PROTODATAViewGenerator(object):
    def __init__(self, num_patch=5):
        self.num_patch = num_patch

    def __call__(self, x):
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        aug_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomResizedCrop(224, scale=(0.9, 0.9), ratio=(1, 1)),  # the scale is critical
            transforms.ToTensor(),
            normalize
        ])
        augmented_x = [aug_transform(x) for i in range(self.num_patch)]

        return augmented_x


class CLIP224QUERYDATAViewGenerator(object):
    def __init__(self, num_patch=5):
        self.num_patch = num_patch

    def __call__(self, x):
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        aug_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize
        ])
        augmented_x = [aug_transform(x) for i in range(self.num_patch)]

        return augmented_x


cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
stl10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
svhn_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
gtsrb_classes = ['Speed limit 20km/h',
                        'Speed limit 30km/h',
                        'Speed limit 50km/h',
                        'Speed limit 60km/h',
                        'Speed limit 70km/h',
                        'Speed limit 80km/h', #5
                        'End of speed limit 80km/h',
                        'Speed limit 100km/h',
                        'Speed limit 120km/h',
                        'No passing sign',
                        'No passing for vehicles over 3.5 metric tons', #10
                        'Right-of-way at the next intersection',
                        'Priority road sign',
                        'Yield sign',
                        'Stop sign', #14
                        'No vehicles sign',  #15
                        'Vehicles over 3.5 metric tons prohibited',
                        'No entry',
                        'General caution',
                        'Dangerous curve to the left',
                        'Dangerous curve to the right', #20
                        'Double curve',
                        'Bumpy road',
                        'Slippery road',
                        'Road narrows on the right',
                        'Road work',    #25
                        'Traffic signals',
                        'Pedestrians crossing',
                        'Children crossing',
                        'Bicycles crossing',
                        'Beware of ice or snow',   #30
                        'Wild animals crossing',
                        'End of all speed and passing limits',
                        'Turn right ahead',
                        'Turn left ahead',
                        'Ahead only',   #35
                        'Go straight or right',
                        'Go straight or left',
                        'Keep right',
                        'Keep left',
                        'Roundabout mandatory', #40
                        'End of no passing',
                        'End of no passing by vehicles over 3.5 metric tons']


def choose_imagenet_file(args):
    if args.query_data_num == 500:
        numpy_file_name = args.query_data_dir + 'surrogate_data_500.npz'
    elif args.query_data_num == 625:
        numpy_file_name = args.query_data_dir + 'surrogate_data_625.npz'
    elif args.query_data_num == 1000:
        numpy_file_name = args.query_data_dir + 'surrogate_data_1000.npz'
    elif args.query_data_num == 1250:
        numpy_file_name = args.query_data_dir + 'surrogate_data_1250.npz'
    elif args.query_data_num == 1500:
        numpy_file_name = args.query_data_dir + 'surrogate_data_1500.npz'
    elif args.query_data_num == 2000:
        numpy_file_name = args.query_data_dir + 'surrogate_data_2000.npz'
    elif args.query_data_num == 2500:
        numpy_file_name = args.query_data_dir + 'surrogate_data_2500.npz'
    elif args.query_data_num == 3000:
        numpy_file_name = args.query_data_dir + 'surrogate_data_3000.npz'
    elif args.query_data_num == 3500:
        numpy_file_name = args.query_data_dir + 'surrogate_data_3500.npz'
    elif args.query_data_num == 4000:
        numpy_file_name = args.query_data_dir + 'surrogate_data_4000.npz'
    elif args.query_data_num == 4500:
        numpy_file_name = args.query_data_dir + 'surrogate_data_4500.npz'
    elif args.query_data_num == 5000:
        numpy_file_name = args.query_data_dir + 'surrogate_data_5000.npz'
    elif args.query_data_num == 10000:
        numpy_file_name = args.query_data_dir + 'surrogate_data_10000.npz'
    elif args.query_data_num == 20000:
        numpy_file_name = args.query_data_dir + 'surrogate_data_20000.npz'
    elif args.query_data_num == 25000:
        numpy_file_name = args.query_data_dir + 'surrogate_data_25000.npz'
    elif args.query_data_num == 40000:
        numpy_file_name = args.query_data_dir + 'surrogate_data_40000.npz'
    elif args.query_data_num == 60000:
        numpy_file_name = args.query_data_dir + 'surrogate_data_60000.npz'
    elif args.query_data_num == 80000:
        numpy_file_name = args.query_data_dir + 'surrogate_data_80000.npz'

    return numpy_file_name


def get_downstream_test_data(args):

    if args.pretraining_dataset == 'cifar10':
        test_transform = test_transform_cifar10
    elif args.pretraining_dataset == 'stl10':
        test_transform = test_transform_stl10
    elif args.pretraining_dataset == 'imagenet':
        test_transform = test_transform_imagenet
    elif args.pretraining_dataset == 'clip':
        test_transform = test_transform_CLIP

    if args.target_downstream_dataset == 'cifar10':
        classes = cifar10_classes
    elif args.target_downstream_dataset == 'stl10':
        classes = stl10_classes
    elif args.target_downstream_dataset == 'gtsrb':
        classes = gtsrb_classes
    elif args.target_downstream_dataset == 'svhn':
        classes = svhn_classes

    memory_data = TESTDATA(numpy_file=args.test_data_dir + "train.npz", class_type=classes, transform=test_transform)
    test_data = TESTDATA(numpy_file=args.test_data_dir + "test.npz", class_type=classes, transform=test_transform)

    return memory_data, test_data

# CIFAR10
def get_query_cifar10(args):
    query_data_num = args.query_data_num
    query_data_sampling_indices = np.random.choice(50000, query_data_num, replace=False)
    query_transform = CIFARQUERYDATAViewGenerator(num_patch=args.query_patch)
    query_data = CIFAR10QUERYDataset(numpy_file=args.query_data_dir+'train.npz', indices=query_data_sampling_indices, transform=query_transform)

    return query_data, query_data_sampling_indices


def get_protos_cifar10(args, query_data_sampling_indices):
    if args.proto_patch != 1:
        proto_transform = CIFARPROTODATAViewGenerator(num_patch=args.proto_patch)
    else:
        proto_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    query_data = CIFAR10QUERYDataset(numpy_file=args.query_data_dir + 'train.npz', indices=query_data_sampling_indices, transform=proto_transform)

    return query_data


def get_query_pair_conventional_cifar10(args):
    query_data_num = args.query_data_num
    query_data_sampling_indices = np.random.choice(50000, query_data_num, replace=False)
    query_data = CIFAR10QUERYPairData(numpy_file=args.query_data_dir+'train.npz', indices=query_data_sampling_indices, transform_1=None, transform_2=None)

    return query_data


def get_query_pair_stolenencoder_cifar10(args):
    query_data_num = args.query_data_num
    query_data_sampling_indices = np.random.choice(50000, query_data_num, replace=False)
    query_data = CIFAR10QUERYPairData(numpy_file=args.query_data_dir+'train.npz', indices=query_data_sampling_indices, transform_1=None, transform_2=query_pair_cifar10_transform)

    return query_data, query_data_sampling_indices


def get_query_pair_consteal_cifar10(args):
    query_data_num = args.query_data_num
    query_data_sampling_indices = np.random.choice(50000, query_data_num, replace=False)
    query_data = CIFAR10QUERYPairData(numpy_file=args.query_data_dir+'train.npz', indices=query_data_sampling_indices, transform_1=query_pair_cifar10_transform, transform_2=query_pair_cifar10_transform)

    return query_data

# STL10
def get_query_stl10(args):
    query_data_num = args.query_data_num
    query_data_sampling_indices = np.random.choice(105000, query_data_num, replace=False)
    query_transform = CIFARQUERYDATAViewGenerator(num_patch=args.query_patch)
    query_data = CIFAR10QUERYDataset(numpy_file=args.query_data_dir + 'train_unlabeled.npz', indices=query_data_sampling_indices, transform=query_transform)

    return query_data, query_data_sampling_indices


def get_protos_stl10(args, query_data_sampling_indices):
    if args.proto_patch != 1:
        proto_transform = CIFARPROTODATAViewGenerator(num_patch=args.proto_patch)
    else:
        proto_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    query_data = CIFAR10QUERYDataset(numpy_file=args.query_data_dir + 'train_unlabeled.npz', indices=query_data_sampling_indices, transform=proto_transform)

    return query_data


def get_query_pair_conventional_stl10(args):
    query_data_num = args.query_data_num
    query_data_sampling_indices = np.random.choice(105000, query_data_num, replace=False)
    query_data = CIFAR10QUERYPairData(numpy_file=args.query_data_dir+'train_unlabeled.npz', indices=query_data_sampling_indices, transform_1=None, transform_2=None)

    return query_data


def get_query_pair_stolenencoder_stl10(args):
    query_data_num = args.query_data_num
    query_data_sampling_indices = np.random.choice(105000, query_data_num, replace=False)
    query_data = CIFAR10QUERYPairData(numpy_file=args.query_data_dir+'train_unlabeled.npz', indices=query_data_sampling_indices, transform_1=None, transform_2=query_pair_cifar10_transform)

    return query_data, query_data_sampling_indices


def get_query_pair_consteal_stl10(args):
    query_data_num = args.query_data_num
    query_data_sampling_indices = np.random.choice(105000, query_data_num, replace=False)
    query_data = CIFAR10QUERYPairData(numpy_file=args.query_data_dir+'train_unlabeled.npz', indices=query_data_sampling_indices, transform_1=query_pair_cifar10_transform, transform_2=query_pair_cifar10_transform)

    return query_data

# SVHN
def get_query_svhn(args):
    query_data_num = args.query_data_num
    query_data_sampling_indices = np.random.choice(73257, query_data_num, replace=False)
    query_transform = CIFARQUERYDATAViewGenerator(num_patch=args.query_patch)
    query_data = CIFAR10QUERYDataset(numpy_file=args.query_data_dir + 'train.npz', indices=query_data_sampling_indices, transform=query_transform)

    return query_data, query_data_sampling_indices


def get_protos_svhn(args, query_data_sampling_indices):
    if args.proto_patch != 1:
        proto_transform = CIFARPROTODATAViewGenerator(num_patch=args.proto_patch)
    else:
        proto_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    query_data = CIFAR10QUERYDataset(numpy_file=args.query_data_dir + 'train.npz', indices=query_data_sampling_indices, transform=proto_transform)

    return query_data


def get_query_pair_conventional_svhn(args):
    query_data_num = args.query_data_num
    query_data_sampling_indices = np.random.choice(73257, query_data_num, replace=False)
    query_data = CIFAR10QUERYPairData(numpy_file=args.query_data_dir+'train.npz', indices=query_data_sampling_indices, transform_1=None, transform_2=None)

    return query_data


def get_query_pair_stolenencoder_svhn(args):
    query_data_num = args.query_data_num
    query_data_sampling_indices = np.random.choice(73257, query_data_num, replace=False)
    query_data = CIFAR10QUERYPairData(numpy_file=args.query_data_dir+'train.npz', indices=query_data_sampling_indices, transform_1=None, transform_2=query_pair_cifar10_transform)

    return query_data, query_data_sampling_indices


def get_query_pair_consteal_svhn(args):
    query_data_num = args.query_data_num
    query_data_sampling_indices = np.random.choice(73257, query_data_num, replace=False)
    query_data = CIFAR10QUERYPairData(numpy_file=args.query_data_dir+'train.npz', indices=query_data_sampling_indices, transform_1=query_pair_cifar10_transform, transform_2=query_pair_cifar10_transform)

    return query_data

# GTSRB
def get_query_gtsrb(args):
    query_data_num = args.query_data_num
    query_data_sampling_indices = np.random.choice(39209, query_data_num, replace=False)
    query_transform = CIFARQUERYDATAViewGenerator(num_patch=args.query_patch)
    query_data = CIFAR10QUERYDataset(numpy_file=args.query_data_dir + 'train.npz', indices=query_data_sampling_indices, transform=query_transform)

    return query_data, query_data_sampling_indices


def get_protos_gtsrb(args, query_data_sampling_indices):
    if args.proto_patch != 1:
        proto_transform = CIFARPROTODATAViewGenerator(num_patch=args.proto_patch)
    else:
        proto_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    query_data = CIFAR10QUERYDataset(numpy_file=args.query_data_dir + 'train.npz', indices=query_data_sampling_indices, transform=proto_transform)

    return query_data


def get_query_pair_conventional_gtsrb(args):
    query_data_num = args.query_data_num
    query_data_sampling_indices = np.random.choice(39209, query_data_num, replace=False)
    query_data = CIFAR10QUERYPairData(numpy_file=args.query_data_dir+'train.npz', indices=query_data_sampling_indices, transform_1=None, transform_2=None)

    return query_data


def get_query_pair_stolenencoder_gtsrb(args):
    query_data_num = args.query_data_num
    query_data_sampling_indices = np.random.choice(39209, query_data_num, replace=False)
    query_data = CIFAR10QUERYPairData(numpy_file=args.query_data_dir+'train.npz', indices=query_data_sampling_indices, transform_1=None, transform_2=query_pair_cifar10_transform)

    return query_data, query_data_sampling_indices


def get_query_pair_consteal_gtsrb(args):
    query_data_num = args.query_data_num
    query_data_sampling_indices = np.random.choice(39209, query_data_num, replace=False)
    query_data = CIFAR10QUERYPairData(numpy_file=args.query_data_dir+'train.npz', indices=query_data_sampling_indices, transform_1=query_pair_cifar10_transform, transform_2=query_pair_cifar10_transform)

    return query_data

# ImageNet
def get_query_imagenet(args):

    numpy_file_name = choose_imagenet_file(args)

    query_data_sampling_indices = np.random.choice(args.query_data_num, args.query_data_num, replace=False)
    query_transform = ImageNetQUERYDATAViewGenerator(num_patch=args.query_patch)
    query_data = CIFAR10QUERYDataset(numpy_file=numpy_file_name, indices=query_data_sampling_indices, transform=query_transform)

    return query_data, query_data_sampling_indices


def get_protos_imagenet(args, query_data_sampling_indices):
    if args.proto_patch != 1:
        proto_transform = ImageNetPROTODATAViewGenerator(num_patch=args.proto_patch)
    else:
        proto_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    numpy_file_name = choose_imagenet_file(args)

    query_data = CIFAR10QUERYDataset(numpy_file=numpy_file_name, indices=query_data_sampling_indices, transform=proto_transform)

    return query_data


def get_query_pair_conventional_imagenet(args):
    numpy_file_name = choose_imagenet_file(args)

    query_data_sampling_indices = np.random.choice(args.query_data_num, args.query_data_num, replace=False)
    query_data = CIFAR10QUERYPairData(numpy_file=numpy_file_name, indices=query_data_sampling_indices, transform_1=None, transform_2=None)

    return query_data


def get_query_pair_stolenencoder_imagenet(args):
    numpy_file_name = choose_imagenet_file(args)

    query_data_sampling_indices = np.random.choice(args.query_data_num, args.query_data_num, replace=False)
    query_data = CIFAR10QUERYPairData(numpy_file=numpy_file_name, indices=query_data_sampling_indices, transform_1=None, transform_2=query_pair_imagenet_transform)

    return query_data, query_data_sampling_indices


def get_query_pair_consteal_imagenet(args):
    numpy_file_name = choose_imagenet_file(args)

    query_data_sampling_indices = np.random.choice(args.query_data_num, args.query_data_num, replace=False)
    query_data = CIFAR10QUERYPairData(numpy_file=numpy_file_name, indices=query_data_sampling_indices, transform_1=query_pair_imagenet_transform, transform_2=query_pair_imagenet_transform)

    return query_data


# ImageNet 224*224
def get_query_imagenet_224(args):
    numpy_file_name = choose_imagenet_file(args)
    query_data_sampling_indices = np.random.choice(args.query_data_num, args.query_data_num, replace=False)

    if args.pretraining_dataset == 'imagenet':
        query_transform = ImageNet224QUERYDATAViewGenerator(num_patch=args.query_patch)
    else:
        query_transform = CLIP224QUERYDATAViewGenerator(num_patch=args.query_patch)

    query_data = CIFAR10QUERYDataset(numpy_file=numpy_file_name, indices=query_data_sampling_indices, transform=query_transform)

    return query_data, query_data_sampling_indices


def get_protos_imagenet_224(args, query_data_sampling_indices):
    if args.pretraining_dataset == 'clip':
        if args.proto_patch != 1:
            proto_transform = CLIP224PROTODATAViewGenerator(num_patch=args.proto_patch)
        else:
            proto_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    else:
        if args.proto_patch != 1:
            proto_transform = ImageNet224PROTODATAViewGenerator(num_patch=args.proto_patch)
        else:
            proto_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])


    numpy_file_name = choose_imagenet_file(args)
    query_data = CIFAR10QUERYDataset(numpy_file=numpy_file_name, indices=query_data_sampling_indices, transform=proto_transform)

    return query_data

# STL10
def get_query_stl10_224(args):
    query_data_num = args.query_data_num
    query_data_sampling_indices = np.random.choice(105000, query_data_num, replace=False)
    if args.pretraining_dataset == 'imagenet':
        query_transform = ImageNet224QUERYDATAViewGenerator(num_patch=args.query_patch)
    else:
        query_transform = CLIP224QUERYDATAViewGenerator(num_patch=args.query_patch)

    query_data = CIFAR10QUERYDataset(numpy_file=args.query_data_dir + 'train_unlabeled.npz', indices=query_data_sampling_indices, transform=query_transform)

    return query_data, query_data_sampling_indices


def get_protos_stl10_224(args, query_data_sampling_indices):
    if args.pretraining_dataset == 'clip':
        if args.proto_patch != 1:
            proto_transform = CLIP224PROTODATAViewGenerator(num_patch=args.proto_patch)
        else:
            proto_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    else:
        if args.proto_patch != 1:
            proto_transform = ImageNet224PROTODATAViewGenerator(num_patch=args.proto_patch)
        else:
            proto_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

    query_data = CIFAR10QUERYDataset(numpy_file=args.query_data_dir + 'train_unlabeled.npz', indices=query_data_sampling_indices, transform=proto_transform)

    return query_data


query_pair_cifar10_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.75, 0.75), ratio=(1, 1)),  # the scale is critical
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    GBlur(p=0.1),
    transforms.RandomApply([Solarization()], p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

query_pair_imagenet_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomResizedCrop(32, scale=(0.75, 0.75), ratio=(1, 1)),  # the scale is critical
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    GBlur(p=0.1),
    transforms.RandomApply([Solarization()], p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])