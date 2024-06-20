import os
import argparse
import random

import torchvision
import numpy as np
from functools import partial
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import get_encoder_architecture, get_downstream_model
from datasets import get_downstream_datasets
from evaluation.nn_classifier import predict_feature, create_torch_dataloader, NeuralNet, net_train, net_test


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate the trained encoders')
    parser.add_argument('--task', default='downstream_task', type=str, help='tasks the file does')
    parser.add_argument('--pretraining_dataset', default='cifar10', type=str, help='dataset the victim uses for pretraining')
    parser.add_argument('--dataset', default='gtsrb', type=str, help='downstream dataset')
    parser.add_argument('--encoder', default='results/saved models/our_model_steal_mobilenet_v2_seed100_imagenet.pth', type=str, help='path to the image encoder')
    parser.add_argument('--gpu', default='0', type=str, help='the index of gpu used to train the model')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--seed', default=50, type=int, help='seed')
    parser.add_argument('--nn_epochs', default=100, type=int)
    parser.add_argument('--hidden_size_1', default=512, type=int)
    parser.add_argument('--hidden_size_2', default=256, type=int)
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='mini-batch size')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args.data_dir = f'./data/{args.dataset}/'

    train_data, test_data = get_downstream_datasets(args)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False,
                              num_workers=12,
                              pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                             num_workers=12,
                             pin_memory=True)

    num_of_classes = len(train_data.classes)

    encoder = get_downstream_model(args).cuda()

    if 'cifar10' in args.encoder or 'stl10' in args.encoder:
        check_point = torch.load(args.encoder)
        encoder.load_state_dict(check_point['state_dict'])
        feature_bank_training, label_bank_training = predict_feature(encoder.f, train_loader)
        feature_bank_testing, label_bank_testing = predict_feature(encoder.f, test_loader)

    elif 'our' in args.encoder or 'Conventional' in args.encoder or 'StolenEncoder' in args.encoder or 'ContSteal' in args.encoder:
        check_point = torch.load(args.encoder)
        encoder.f.load_state_dict(check_point['state_dict'])
        feature_bank_training, label_bank_training = predict_feature(encoder.f, train_loader)
        feature_bank_testing, label_bank_testing = predict_feature(encoder.f, test_loader)

    else:
        if args.pretraining_dataset in ['imagenet', 'clip']:
            check_point = torch.load(args.encoder)
            state_dict = check_point['state_dict']
            encoder.visual.load_state_dict(state_dict)
            feature_bank_training, label_bank_training = predict_feature(encoder.visual, train_loader)
            feature_bank_testing, label_bank_testing = predict_feature(encoder.visual, test_loader)

    nn_train_loader = create_torch_dataloader(feature_bank_training, label_bank_training, args.batch_size)
    nn_test_loader = create_torch_dataloader(feature_bank_testing, label_bank_testing, args.batch_size)

    input_size = feature_bank_training.shape[1]
    criterion = nn.CrossEntropyLoss()

    classifier = NeuralNet(input_size, [args.hidden_size_1, args.hidden_size_2], num_of_classes).cuda()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)

    for epoch in range(1, args.nn_epochs + 1):
        net_train(classifier, nn_train_loader, optimizer, epoch, criterion)
        net_test(classifier, nn_test_loader, epoch, criterion, 'Accuracy')
