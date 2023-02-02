#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time 
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details
from deep_learning import NeuralNet
import pickle

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('')

    args = args_parser()
    exp_details(args)

    #if args.gpu_id:
    #    torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    #client별 data 개수 배열
    local_data = [len(data) for user, data in user_groups.items()]
    client_id = [user for user in user_groups.keys()]
    
    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural network
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
        
    elif args.model == 'color':
        input_size = 92 # n features
        num_output = 31
        hidden_sizes = [600]
        global_model = NeuralNet(input_size, hidden_sizes, num_output)
        
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    #global_model.to(device)
    global_model.train()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    print(global_model)

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        #m = 1
        idxs_users = np.random.choice(client_id, m, replace=False)
        local_weights = [0 for _ in user_groups.keys()]
        
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,idxs=user_groups[idx],
                                       logger=logger, input_size=input_size, output_size=num_output)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights[client_id.index(idx)] = copy.deepcopy(w)
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(global_model.state_dict(), local_weights, local_data)
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

    # Test inference after completion of training
    test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Loss: {:.2f}%".format(train_loss[-1]))

    # Saving the objects train_loss and train_accuracy:
    file_name = './save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    logger.close()

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
