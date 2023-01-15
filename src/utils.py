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


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
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

        dict_users = {}
        #for person in df['REGUSER'].unique():
        for person in [30201, 21907011]:
            dict_users[person] =  train_dataset[train_dataset['REGUSER'] == person].index.tolist()
        
    return train_dataset, test_dataset, dict_users


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    print(w_avg.keys())
    for key in w_avg.keys(): #각 LAYER에 대해 실행
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
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
