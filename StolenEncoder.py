import copy

import torch
import argparse
import random
import os
import json
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from models import get_encoder_architecture, get_surrogate_model
from datasets import get_query_dataset, get_pretraining_dataset, get_proto_dataset, get_downstream_test_data, get_pair_query_stolenencoder
from evaluation import knn_predict
from colorama import init,Fore
import time

init(autoreset=True)


def chunk_avg(x, n_chunks=2, normalize=False):
    x_list = x.chunk(n_chunks, dim=0)
    x = torch.stack(x_list, dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0),dim=1)

def steal_train(args, query_data_loader, prototype, surrogate_model, steal_optimizer, loss_func, epoch):
    surrogate_model.train()
    loss_function = loss_func
    total_num = 0
    total_loss = 0

    for batch, (img_1, img_2, label) in tqdm(enumerate(query_data_loader), desc='Training process of epoch [{}/{}]'.format(epoch, args.epochs)):
        img_1 = img_1.cuda()
        img_2 = img_2.cuda()
        surrogate_embeddings = F.normalize(surrogate_model(img_1), dim=-1)
        surrogate_embeddings_aug = F.normalize(surrogate_model(img_2), dim=-1)

        for i, l in enumerate(label):
            if i == 0:
                protos = torch.unsqueeze(prototype[int(l)], dim=0)
            else:
                protos = torch.cat((protos, torch.unsqueeze(prototype[int(l)], dim=0)))

        loss = 0
        if args.loss_function == 'mse':
            loss += loss_function(torch.squeeze(surrogate_embeddings), protos.to('cuda'))
            loss += 20*loss_function(torch.squeeze(surrogate_embeddings_aug), protos.to('cuda'))
        elif args.loss_function == 'cos_similarity':
            loss -= torch.sum(loss_function(torch.squeeze(surrogate_embeddings), protos.to('cuda'))) / args.batch_size
            loss -= torch.sum(loss_function(torch.squeeze(surrogate_embeddings_aug), protos.to('cuda'))) / args.batch_size
        elif args.loss_function == 'cross_entropy':
            loss += loss_function(torch.squeeze(surrogate_embeddings), protos.to('cuda'))
            loss += loss_function(torch.squeeze(surrogate_embeddings_aug), protos.to('cuda'))

        Loss = loss
        steal_optimizer.zero_grad()
        Loss.backward()
        steal_optimizer.step()
        total_num += query_data_loader.batch_size
        total_loss += Loss.item() * query_data_loader.batch_size
    avg_loss = total_loss / total_num
    print('Training loss of epoch {}: {}'.format(epoch, avg_loss), flush=True)
    return surrogate_model

def test(net, memory_data_loader, test_data_clean_loader, epoch, args):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_num, feature_bank = 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net(data.cuda(non_blocking=True))
            feature = torch.squeeze(feature)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_clean_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = torch.squeeze(feature)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))

    return total_top1 / total_num * 100

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Steal the pretrained encoder')

    parser.add_argument('--task', default='steal', type=str, help='tasks the file does')
    parser.add_argument('--pretraining_dataset', default='cifar10', type=str, help='dataset the victim uses for pretraining')
    parser.add_argument('--proto_batch_size', default=100, type=int, help='image size in each batch of prototype data')
    parser.add_argument('--query_batch_size', default=100, type=int, help='image size in each batch of query data')
    parser.add_argument('--query_dataset', default='imagenet', type=str, help='dataset used for querying the victim', choices=['cifar10', 'stl10', 'svhn', 'gtsrb'])
    parser.add_argument('--proto_patch', default=1, type=int, help='patches used for generating prototypical representations')
    parser.add_argument('--query_data_num', default=2500, type=int, help='how much query data we use')
    parser.add_argument('--target_downstream_dataset', default='cifar10', type=str, help='target downstream dataset')
    parser.add_argument('--target_check_point', default='results/cifar10_simclr_model_1000.pth', type=str, help='path to the check point of the target model')
    parser.add_argument('--architecture', default='resnet18', type=str, help='surrogate model architecture')
    parser.add_argument('--loss_function', default='mse', type=str, help='loss function used for attacker', choices=['mse', 'cos_similarity', 'cross_entropy', 'ours'])
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate in Adam optimizer')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to steal the target model')
    parser.add_argument('--test_freq', default=1, type=int, help='the frequency to test the model')
    parser.add_argument('--model_results_dir', default='results/saved models', type=str, help='path to save models')
    parser.add_argument('--log_results_dir', default='results/logs', type=str, help='path to save logs')
    parser.add_argument('--knn-t', default=0.5, type=float, help='softmax temperature in kNN monitor')
    parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')
    parser.add_argument('--seed', default=100, type=int, help='which seed the code runs on')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu the code runs on')
    CUDA_LAUNCH_BLOCKING=1
    args = parser.parse_args()

    # Set the random seeds and GPU information
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # load target/victim model
    if args.pretraining_dataset == 'cifar10' or args.pretraining_dataset == 'stl10':
        target_model = get_encoder_architecture(args).cuda()
        target_check_point = torch.load(args.target_check_point)
        target_model.load_state_dict(target_check_point['state_dict'])
        target_model = target_model.f
    elif args.pretraining_dataset == 'imagenet':
        target_model = get_encoder_architecture(args).cuda()
        checkpoint = torch.load(args.target_check_point)
        target_model.visual.load_state_dict(checkpoint['state_dict'])


    # initialize surrogate model
    surrogate_model = get_surrogate_model(args).cuda()

    # load the query dataset and proto dataset
    args.query_data_dir = f'./data/{args.query_dataset}/query/'
    args.test_data_dir = f'./data/{args.target_downstream_dataset}/'

    query_dataset, query_data_sampling_indices = get_pair_query_stolenencoder(args)
    proto_dataset = get_proto_dataset(args, query_data_sampling_indices)
    memory_data, test_data = get_downstream_test_data(args)

    proto_data_loader = DataLoader(proto_dataset, batch_size=args.proto_batch_size, shuffle=False,
                                   num_workers=12,
                                   pin_memory=True)

    query_data_loader = DataLoader(query_dataset, batch_size=args.query_batch_size, shuffle=True,
                                   num_workers=12,
                                   pin_memory=True, drop_last=True)

    memory_loader = DataLoader(memory_data, batch_size=args.proto_batch_size, shuffle=False,
                               num_workers=12,
                               pin_memory=True)

    test_loader = DataLoader(test_data, batch_size=args.proto_batch_size, shuffle=False,
                             num_workers=12,
                             pin_memory=True)

    # test the victim model
    # if args.pretraining_dataset == 'imagenet':
    #     _ = test(target_model.visual, memory_loader, test_loader, 0, args)
    # else:
    #     _ = test(target_model, memory_loader, test_loader, 0, args)  # CIFAR10 Pre-trained: CIFAR10: 88.31%  STL10: 71.59%  GTSRB: 61.3%  SVHN: 40.77%
    # test the initial surrogate model
    # _ = test(surrogate_model, memory_loader, test_loader, 0, args) # CIFAR10: 35.62%  GTSRB: 25.91%  SVHN: 26.56%


    # Extracting prototypical representations of the victim encoder
    prototypes_bank = dict()
    target_model.eval()
    begin_time = time.time()
    with torch.no_grad():
        if args.proto_patch != 1:
            for (data, labels) in tqdm(proto_data_loader, desc='Prototypical feature extraction'):
                data = torch.cat(data, dim=0)
                data = data.cuda()
                target_embeddings = F.normalize(target_model(data), dim=-1).detach().cpu()
                target_embedding_avg = torch.squeeze(chunk_avg(target_embeddings, args.proto_patch))
                for index, label in enumerate(labels):
                    prototypes_bank[int(label)] = target_embedding_avg[index]

        else:
            for (data, labels) in tqdm(proto_data_loader, desc='Prototypical feature extraction'):
                data = data.cuda()
                target_embeddings = F.normalize(target_model(data), dim=-1).detach().cpu()
                target_embedding_avg = torch.squeeze(target_embeddings)
                for index, label in enumerate(labels):
                    prototypes_bank[int(label)] = target_embedding_avg[index]
    end_time = time.time()
    print('Extracting prototypical representations takes {:.2f} seconds'.format(end_time-begin_time), flush=True)


    # determine the loss function for training
    if args.loss_function == 'mse':
        criterion = nn.MSELoss().to('cuda')
    elif args.loss_function == 'cos_similarity':
        criterion = nn.CosineSimilarity(dim=-1).to('cuda')
    elif args.loss_function == 'cross_entropy':
        criterion = nn.CrossEntropyLoss().to('cuda')

    steal_optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Logging
    results = {'SA': []}
    if not os.path.exists(args.log_results_dir):
        os.mkdir(args.log_results_dir)

    # Dump args
    with open(args.log_results_dir + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)


    print('='*15,'Steal the Prototypical Representations of Self-Supervised Encoders','='*15, flush=True)
    epoch_start = 1
    best_acc = 0
    for epoch in range(epoch_start, args.epochs+1):
        if epoch == epoch_start:
            begin_time = time.time()

        surrogate_model_new = steal_train(args, query_data_loader, prototypes_bank, surrogate_model, steal_optimizer, criterion, epoch)

        if epoch == 10:
            end_time = time.time()
            print('On average, 1 training epoch takes {:.2f} seconds.'.format((end_time - begin_time) / 10), flush=True)

        if epoch % args.test_freq == 0:
            test_acc = test(surrogate_model_new, memory_loader, test_loader, epoch, args)
            if test_acc >= best_acc:
                results['SA'].append(test_acc)
            else:
                results['SA'].append(best_acc)
            # Save statistics
            data_frame = pd.DataFrame(data=results, index=range(args.test_freq, epoch+args.test_freq, args.test_freq))
            data_frame.to_csv(args.log_results_dir + '/StolenEncoder_log_seed{}.csv'.format(args.seed), index_label='epoch')
            if test_acc >= best_acc:
                best_acc = test_acc
                torch.save({'state_dict': surrogate_model_new.state_dict()}, args.model_results_dir + '/StolenEncoder_seed{}_'.format(args.seed)+ args.query_dataset + '.pth')
            print(Fore.BLUE + 'The current best acc:  {:.2f}%'.format(best_acc))