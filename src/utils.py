#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
import os
import pandas as pd
import numpy as np
from collections import OrderedDict

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    user group: user number가 key, user별 data index가 value인 dict
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)
    
    elif args.dataset == 'color':
        
        currentPath = os.getcwd()
        data_location = currentPath + "\\data" 
        df = pd.read_csv(data_location + '\\df_new_norm.csv')
        # train test split
        train_dataset = pd.DataFrame()
        test_dataset = pd.DataFrame()

        for person in [30201, 21907011]:
            
            df_per_user = df[df['REGUSER'] == person]
            
            # train, test data 개수 구하기
            train_count = int(len(df_per_user)*0.9)
            
            #train, test dataset 구축
            train_dataset = pd.concat([train_dataset, df_per_user.iloc[:train_count,:]])
            test_dataset = pd.concat([test_dataset, df_per_user.iloc[train_count:,:]])

        dict_users = OrderedDict()
        #for person in df['REGUSER'].unique():
        for person in [30201, 21907011]:
            dict_users[person] =  train_dataset[train_dataset['REGUSER'] == person].index.tolist()
        
    return train_dataset, test_dataset, dict_users


def average_weights(global_weight, local_weights, local_data):
    """
    Returns the average of the weights.
    local_weights: 전체 클라이언트의 weight 배열, 이번에 업데이트 진행 X라면 0
    """
    w_avg = copy.deepcopy(global_weight)
    total_data = sum(local_data)
    for key in w_avg.keys(): #각 LAYER에 대해
        for i in range(0, len(local_data)):#각 client에 대해
            if i == 0:
                w_avg[key] = 0
            if local_weights[i] == 0: #이번 라운드 업데이트 X인 클라이언트
                 w_avg[key] += global_weight[key] * (local_data[i]/total_data)
            else:
                w_avg[key] += local_weights[i][key] * (local_data[i]/total_data)
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
