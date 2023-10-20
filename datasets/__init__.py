from .cifar10_dataset import get_pretraining_cifar10, get_downstream_cifar10, get_backdoor_cifar10
from .stl10_dataset import get_pretraining_stl10, get_downstream_stl10
from .gtsrb_dataset import get_downstream_gtsrb
from .stl10_dataset import get_downstream_stl10
from .svhn_dataset import get_pretraining_svhn, get_downstream_svhn
from .mnist_dataset import get_downstream_mnist
from .fmnist_dataset import get_downstream_fmnist
from .cifar100_dataset import get_downstream_cifar100
from .query_dataset import get_query_cifar10, get_protos_cifar10, get_query_pair_conventional_cifar10, get_query_pair_stolenencoder_cifar10, get_query_pair_consteal_cifar10,\
    get_query_stl10, get_protos_stl10, get_query_pair_conventional_stl10, get_query_pair_stolenencoder_stl10, get_query_pair_consteal_stl10,\
    get_query_svhn, get_protos_svhn, get_query_pair_conventional_svhn, get_query_pair_stolenencoder_svhn, get_query_pair_consteal_svhn,\
    get_query_gtsrb, get_protos_gtsrb, get_query_pair_conventional_gtsrb, get_query_pair_stolenencoder_gtsrb, get_query_pair_consteal_gtsrb,\
    get_query_imagenet, get_protos_imagenet, get_query_pair_conventional_imagenet, get_query_pair_stolenencoder_imagenet, get_query_pair_consteal_imagenet,\
    get_query_imagenet_224, get_protos_imagenet_224, get_downstream_test_data, get_protos_stl10_224, get_query_stl10_224


def get_pretraining_dataset(args):
    if args.pretraining_dataset == 'cifar10':
        return get_pretraining_cifar10(args.data_dir)
    if args.pretraining_dataset == 'stl10':
        return get_pretraining_stl10(args.data_dir)
    else:
        raise NotImplementedError

def get_query_dataset(args):
    if args.pretraining_dataset not in ['imagenet', 'clip']:
        if args.query_dataset == 'cifar10':
            return get_query_cifar10(args)
        elif args.query_dataset == 'stl10':
            return get_query_stl10(args)
        elif args.query_dataset == 'svhn':
            return get_query_svhn(args)
        elif args.query_dataset == 'gtsrb':
            return get_query_gtsrb(args)
        elif args.query_dataset == 'imagenet':
            return get_query_imagenet(args)
    else:
        if args.query_dataset == 'imagenet':
            return get_query_imagenet_224(args)
        elif args.query_dataset == 'stl10':
            return get_query_stl10_224(args)

def get_proto_dataset(args, query_data_sampling_indices):
    if args.pretraining_dataset not in ['imagenet', 'clip']:
        if args.query_dataset == 'cifar10':
            return get_protos_cifar10(args, query_data_sampling_indices)
        elif args.query_dataset == 'stl10':
            return get_protos_stl10(args, query_data_sampling_indices)
        elif args.query_dataset == 'svhn':
            return get_protos_svhn(args, query_data_sampling_indices)
        elif args.query_dataset == 'gtsrb':
            return get_protos_gtsrb(args, query_data_sampling_indices)
        elif args.query_dataset == 'imagenet':
            return get_protos_imagenet(args, query_data_sampling_indices)
    else:
        if args.query_dataset == 'imagenet':
            return get_protos_imagenet_224(args, query_data_sampling_indices)
        elif args.query_dataset == 'stl10':
            return get_protos_stl10_224(args, query_data_sampling_indices)

def get_pair_query_conventional(args):
    if args.pretraining_dataset != 'imagenet':
        if args.query_dataset == 'cifar10':
            return get_query_pair_conventional_cifar10(args)
        elif args.query_dataset == 'stl10':
            return get_query_pair_conventional_stl10(args)
        elif args.query_dataset == 'svhn':
            return get_query_pair_conventional_svhn(args)
        elif args.query_dataset == 'gtsrb':
            return get_query_pair_conventional_gtsrb(args)
        elif args.query_dataset == 'imagenet':
            return get_query_pair_conventional_imagenet(args)

def get_pair_query_stolenencoder(args):
    if args.pretraining_dataset != 'imagenet':
        if args.query_dataset == 'cifar10':
            return get_query_pair_stolenencoder_cifar10(args)
        elif args.query_dataset == 'stl10':
            return get_query_pair_stolenencoder_stl10(args)
        elif args.query_dataset == 'svhn':
            return get_query_pair_stolenencoder_svhn(args)
        elif args.query_dataset == 'gtsrb':
            return get_query_pair_stolenencoder_gtsrb(args)
        elif args.query_dataset == 'imagenet':
            return get_query_pair_stolenencoder_imagenet(args)

def get_pair_query_consteal(args):
    if args.pretraining_dataset != 'imagenet':
        if args.query_dataset == 'cifar10':
            return get_query_pair_consteal_cifar10(args)
        elif args.query_dataset == 'stl10':
            return get_query_pair_consteal_stl10(args)
        elif args.query_dataset == 'svhn':
            return get_query_pair_consteal_svhn(args)
        elif args.query_dataset == 'gtsrb':
            return get_query_pair_consteal_gtsrb(args)
        elif args.query_dataset == 'imagenet':
            return get_query_pair_consteal_imagenet(args)

def get_test_datasets(args):
    return get_downstream_test_data(args)

def get_downstream_datasets(args):
    if args.dataset == 'cifar10':
        return get_downstream_cifar10(args)
    elif args.dataset == 'gtsrb':
        return get_downstream_gtsrb(args)
    elif args.dataset == 'stl10':
        return get_downstream_stl10(args)
    elif args.dataset == 'svhn':
        return get_downstream_svhn(args)
    elif args.dataset == 'mnist':
        return get_downstream_mnist(args)
    elif args.dataset == 'fmnist':
        return get_downstream_fmnist(args)
    elif args.dataset == 'cifar100':
        return get_downstream_cifar100(args)


def get_backdoored_data(args):
    return get_backdoor_cifar10(args)