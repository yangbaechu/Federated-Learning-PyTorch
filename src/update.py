#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        measure_spectral = self.dataset.columns.slice_indexer('SPECR_01', 'SPECR_31')
        paper_density = self.dataset.columns.slice_indexer('P_DENSITY', 'P_DENSITY')  # wdw only     (3)
        paper_spectral = self.dataset.columns.slice_indexer('PAPER_SPECR_01','PAPER_SPECR_31')  # paper spectral only (31)
        inks_combination = self.dataset.columns.slice_indexer('20120002','123')  # inks only (56)

        Y = self.dataset.iloc[:, measure_spectral]  # Y input
        X = self.dataset.iloc[:, np.r_[paper_density, paper_spectral, inks_combination]]  # normalized X input
        
        return torch.tensor(X.iloc[item,:].values), torch.tensor(Y.iloc[item,:].values)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to MSELoss loss function
        self.criterion = nn.MSELoss()#.to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, Y) in enumerate(self.trainloader):
                #images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                X = X.to(torch.float32)
                Y_predicted = model(X)

                Y_predicted = Y_predicted.to(torch.float32)
                Y = Y.to(torch.float32)
                loss = self.criterion(Y_predicted, Y)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(X),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item(), global_round)
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (X, Y) in enumerate(self.testloader):
            #images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(X)
            batch_loss = self.criterion(outputs, Y)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, Y)).item()
            total += len(Y)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.MSELoss()#.to(device)
    testloader = DataLoader(DatasetSplit(test_dataset, [i for i in range(len(test_dataset))]), batch_size=128,
                            shuffle=False)
    
    for batch_idx, (X, Y) in enumerate(testloader):
        #images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(X.to(torch.float32))
        batch_loss = criterion(outputs.to(torch.float32),  Y.to(torch.float32))
        loss += batch_loss.item()
        
        test_loss = loss/len(testloader)
        print(test_loss)
    return test_loss
