#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch import nn
from collections import defaultdict
from models import MLP, CNNMnist, CNNCifar, Cifar10CNN, AlexNet, AlexNetMnist
from othermodels.googlenet import googlenet
from othermodels.resnet import resnet18, resnet50
from othermodels.vgg import vgg16
from othermodels.mobilenetv2 import mobilenetv2

from mnistmodels.googlenet import mgooglenet 
from mnistmodels.mobilenetv2 import mmobilenetv2
from mnistmodels.resnet import mresnet50
from mnistmodels.vgg import mvgg16

from global_sets import MODEL_NAME
from label_flipping_attack import *

def get_dataset(dataset):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if dataset == 'cifar':
        data_dir = '../data/cifar/'
        if not os.path.exists(data_dir):
                os.makedirs(data_dir)
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=False,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=False,
                                      transform=apply_transform)

    elif dataset == 'cifar100':
        data_dir = '../data/cifar100/'
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_train)
        # trainset, valset, _ = random_split(trainset, [config.n_train // 2, config.n_train // 2, 50000 - config.n_train])
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform_test)

    elif dataset == 'mnist' or 'fmnist':
        if dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'
        if not os.path.exists(data_dir):
                os.makedirs(data_dir)
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=False,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=False,
                                      transform=apply_transform)

    return train_dataset, test_dataset


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

# def inference(net_g, dataset, batchSize):
#     net_g.eval()
#     # testing
#     test_loss = 0
#     correct = 0
#     data_loader = DataLoader(dataset, batch_size=batchSize.bs)
#     l = len(data_loader)
#     for idx, (data, target) in enumerate(data_loader):
#         data, target = data, target
#         log_probs = net_g(data)
#         # sum up batch loss
#         test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
#         # get the index of the max log-probability
#         y_pred = log_probs.data.max(1, keepdim=True)[1]
#         correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
#
#     test_loss /= len(data_loader.dataset)
#     accuracy = 100.00 * correct / len(data_loader.dataset)
#     # if args.verbose:
#     #     print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
#     #         test_loss, correct, len(data_loader.dataset), accuracy))
#     return accuracy     #, test_loss

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def local_train(self, net, dataset=None, idxs=None, is_flapping_attack=False):
    # device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device=torch.device('cpu')
    net.to(device)
    net.train()

    # train and update
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
    ldr_train = DataLoader(DatasetSplit(dataset, list(idxs)), batch_size=32, shuffle=True)
    epoch_loss = []
    for iter in range(10):
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(device), labels.to(device)
            net.zero_grad()
            outputs = net(images)
            # flapping 攻击
            if is_flapping_attack:
                labels = replace_2_with_7_3_with_7(labels)
            # loss = self.loss_func(log_probs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # if self.args.verbose and batch_idx % 10 == 0:
            #     print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         iter, batch_idx * len(images), len(self.ldr_train.dataset),
            #                 100. * batch_idx / len(self.ldr_train), loss.item()))
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
    return net.state_dict() # , sum(epoch_loss) / len(epoch_loss)

# def update_weights(self, model, global_round):
#     # Set mode to train model
#     model.train()
#     epoch_loss = []
#
#     # Set optimizer for the local updates
#     if self.args.optimizer == 'sgd':
#         optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
#                                         momentum=0.5)
#     elif self.args.optimizer == 'adam':
#         optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
#                                          weight_decay=1e-4)
#
#     for iter in range(self.args.local_ep):
#         batch_loss = []
#         for batch_idx, (images, labels) in enumerate(self.trainloader):
#             images, labels = images.to(self.device), labels.to(self.device)
#
#             model.zero_grad()
#             log_probs = model(images)
#             loss = self.criterion(log_probs, labels)
#             loss.backward()
#             optimizer.step()
#
#             if self.args.verbose and (batch_idx % 10 == 0):
#                 print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                     global_round, iter, batch_idx * len(images),
#                     len(self.trainloader.dataset),
#                         100. * batch_idx / len(self.trainloader), loss.item()))
#             self.logger.add_scalar('loss', loss.item())
#             batch_loss.append(loss.item())
#         epoch_loss.append(sum(batch_loss)/len(batch_loss))
#
#     return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

def getKrum(input):
    '''
    compute krum or multi-krum of input. O(dn^2)

    input : batchsize* vector dimension * n

    return
        krum : batchsize* vector dimension * 1
        mkrum : batchsize* vector dimension * 1
    '''

    n = input.shape[-1]
    f = n // 2  # worse case 50% malicious points
    k = n - f - 2

    # collection distance, distance from points to points
    x = input.permute(0, 2, 1)
    cdist = torch.cdist(x, x, p=2)
    # find the k+1 nbh of each point
    nbhDist, nbh = torch.topk(cdist, k + 1, largest=False)
    # the point closest to its nbh
    i_star = torch.argmin(nbhDist.sum(2))
    # krum
    krum = input[:, :, [i_star]]
    # Multi-Krum
    mkrum = input[:, :, nbh[:, i_star, :].view(-1)].mean(2, keepdims=True)
    return krum, mkrum

def test_inference(model, dataset=None, idxs=None):
    """ Returns the test accuracy and loss.
    """

    # model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    # device=torch.device("cuda:0")
    device = 'cpu' #if args.gpu else 'cpu'
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    if idxs != None:
        ldr_test = DataLoader(DatasetSplit(dataset, list(idxs)), batch_size=256, shuffle=False)
    else:
        ldr_test = DataLoader(dataset, batch_size=256, shuffle=False)

    for batch_idx, (images, labels) in enumerate(ldr_test):
        #model.zero_grad()
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)

        # Prediction
        _, pred_labels = torch.max(outputs.data, 1)
        #pred_labels = pred_labels.view(-1)
        correct += (pred_labels == labels).sum().item()
        total += len(labels)

    accuracy = round(correct / total, 4)
    return accuracy # , loss

def krum_distance(localModels, peers):
    distances = defaultdict(dict)
    for i in range(peers):
        if localModels[i+1] == None:
            continue
        # print('model: ', i+1)
        distances[i+1][i+1] = 1e8
        a = choice_model(MODEL_NAME)
        a.load_state_dict(localModels[i+1].state_dict())
        a_flatten = np.concatenate([param.data.cpu().numpy().flatten() for param in a.parameters()])
        for j in range(i+1, peers):
            if localModels[j+1] == None:
                continue
            b = choice_model(MODEL_NAME)
            b.load_state_dict(localModels[j+1].state_dict())
            b_flatten = np.concatenate([param.data.cpu().numpy().flatten() for param in b.parameters()])
            distances[i+1][j+1] = distances[j+1][i+1] = np.linalg.norm(a_flatten - b_flatten)

    return distances

def krum(user_count, f, distances):
    non_malicious_count = user_count - f - 2
    # distances = krum_distance(localModels)
    mini_error = 1e20
    mini_index = -1
    scores = {}
    for user in distances.keys():
        errors = sorted(distances[user].values())
        current_error = sum(errors[:non_malicious_count])
        scores[user] = 1/current_error
        if current_error < mini_error:
            mini_error = current_error
            mini_index = user

    return mini_index, scores

def multi_krum(localModels, user_count, f, peers):
    distances = krum_distance(localModels, peers)
    sel_set = []
    setsize = user_count - f
    while len(sel_set) < setsize:
        currently_selected, _ = krum(user_count, f, distances)
        sel_set.append(currently_selected)

#         for i in distances.keys():
#             distances[i][currently_selected] = distances[currently_selected][i] = 1e8
        distances.pop(currently_selected)
    return sel_set

def choice_model(model='MLP'):
    if model == 'MLP':
        return MLP()
    elif model == 'net':
        # return CNNCifar()
        # return CNNCifar1()
        return Cifar10CNN()
        #return MLP()
    elif model == 'resnet18':
        return resnet18()
    elif model == 'resnet50':
        return resnet50()
    elif model == 'google':
        return googlenet()
    elif model == 'alex':
        return AlexNet()
    elif model == 'vgg16':
        return vgg16()
    elif model == 'mobile':
        return mobilenetv2()
    ## MNIST dataset
    elif model == 'mresnet50':
        return mresnet50()
    elif model == 'mgoogle':
        return mgooglenet()
    elif model == 'malex':
        return AlexNetMnist()
    elif model == 'mvgg16':
        return mvgg16()
    elif model == 'mmobile':
        return mmobilenetv2()
    
    # elif model == 'net':
    #     return Net()
