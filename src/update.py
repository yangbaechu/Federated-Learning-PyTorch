#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

def get_grad_vector(pp, grad_dims):
    """
    gather the gradients in one vector
    """
    grads = torch.Tensor(sum(grad_dims))
    grads.fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1
    return grads


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
    def __init__(self, args, dataset, idxs, logger, input_size):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to MSELoss loss function
        self.criterion = nn.MSELoss()#.to(self.device)
        
        # continual learning
        self.number_to_select = args.memory_strength
        self.n_memories = args.n_memories
        self.n_sampled_memories = args.n_sampled_memories
        self.n_constraints = args.n_constraints
        
        self.added_index = self.n_sampled_memories
        self.memory_data = torch.FloatTensor(self.n_memories, input_size)
        self.memory_labs = torch.LongTensor(self.n_memories)
        
        # allocate  buffer
        self.sampled_memory_data = None
        self.sampled_memory_labs = None
        self.sampled_memory_cos = None # buffer cosine similarity score
        self.subselect=args.subselect
        
        self.mem_cnt = 0
        
    def cosine_similarity(self, x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)

        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        sim= torch.mm(x1, x2.t())/(w1 * w2.t()) #, w1  # .clamp(min=eps), 1/cosinesim

        return sim
    
    def get_each_batch_sample_sim(self):
        cosine_sim = torch.zeros(self.memory_labs.size(0))
        item_index=0

        for x, y in zip(self.memory_data, self.memory_labs):
            self.zero_grad()
            ptloss = self.ce(self.forward(x.unsqueeze(0)), y.unsqueeze(0))
            ptloss.backward()
            # add the new grad to the memory grads and add it is cosine similarity
            this_grad = get_grad_vector(self.parameters, self.grad_dims).unsqueeze(0)

            cosine_sim[item_index]=max(self.cosine_similarity(self.mem_grads, this_grad))
            item_index+=1

        return cosine_sim


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
        
        self.grad_dims = []
        for param in model.parameters():
            self.grad_dims.append(param.data.numel())

        if self.sampled_memory_data is not None:
            #shuffle buffer, determine batch size of buffer sampled memories
            shuffeled_inds=torch.randperm(self.sampled_memory_labs.size(0))
            effective_batch_size=min(self.n_constraints,self.sampled_memory_labs.size(0))
            b_index=0
        
        #gradients of used buffer samples
        self.mem_grads = None
        
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

                model.zero_grad()
                X, Y = X.to(torch.float32), Y.to(torch.float32)
                Y_predicted = model(X).to(torch.float32)
                
                loss = self.criterion(Y_predicted, Y)
                loss.backward()
                optimizer.step()

                # update steps on the replayed sampels from buffer, we only draw once
                if self.sampled_memory_data is not None:

                    #print(random_batch_inds)
                    random_batch_inds = shuffeled_inds[ b_index * effective_batch_size:b_index * effective_batch_size + effective_batch_size]
                    batch_x=self.sampled_memory_data[random_batch_inds]
                    batch_y = self.sampled_memory_labs[random_batch_inds]
                    self.zero_grad()

                    loss = self.ce(self.forward(batch_x), batch_y)
                    loss.backward()

                    self.opt.step()
                    b_index += 1
                    if b_index * effective_batch_size >= self.sampled_memory_labs.size(0):
                        b_index = 0

                ##HERE MEMORY IS EQUAL TO THE BATCH SIZE, this procedure is performed for every recieved batch
                if self.mem_cnt == self.n_memories :
                    model.eval()

                    if self.sampled_memory_data is not None and self.n_sampled_memories<=self.sampled_memory_data.size(0):#buffer is full


                        batch_sim=self.get_batch_sim(effective_batch_size)#estimate similarity score for the recieved samples to randomly drawn samples from buffer
                        # for effecency we estimate the similarity for the whole batch

                        if (batch_sim)<self.sim_th:


                            mem_data=X.clone()
                            mem_lab=Y.clone()


                            buffer_sim = (self.sampled_memory_cos - torch.min(self.sampled_memory_cos)) / ((torch.max(self.sampled_memory_cos) - torch.min(self.sampled_memory_cos)) + 0.01)

                            index=torch.multinomial(buffer_sim, mem_data.size(0), replacement=False)#draw candidates for replacement from the buffer

                            batch_item_sim=self.get_each_batch_sample_sim()# estimate the similarity of each sample in the recieved batch to the randomly drawn samples from the buffer.
                            scaled_batch_item_sim=((batch_item_sim+1)/2).unsqueeze(1).clone()
                            buffer_repl_batch_sim=((self.sampled_memory_cos[index]+1)/2).unsqueeze(1).clone()
                            #draw an event to decide on replacement decision
                            outcome=torch.multinomial(torch.cat((scaled_batch_item_sim,buffer_repl_batch_sim),dim=1), 1, replacement=False)#
                            #replace samples with outcome =1
                            added_indx = torch.arange(end=batch_item_sim.size(0))
                            sub_index=outcome.squeeze(1).byte()
                            self.sampled_memory_data[index[sub_index]] = mem_data[added_indx[sub_index]].clone()
                            self.sampled_memory_labs[index[sub_index]] = mem_lab[added_indx[sub_index]].clone()

                            self.sampled_memory_cos[index[sub_index]] = batch_item_sim[added_indx[sub_index]].clone()
                
                    else:
                        #add new samples to the buffer
                        added_inds = torch.arange(0, self.memory_data.size(0))

                        #first buffer insertion
                        if self.sampled_memory_data is None:

                            self.sampled_memory_data = self.memory_data[added_inds].clone()
                            self.sampled_memory_labs = self.memory_labs[added_inds].clone()

                            self.sampled_memory_cos=torch.zeros(added_inds.size(0)) + 0.1
                        else:
                            self.get_batch_sim(effective_batch_size)#draw random samples from buffer
                            this_sampled_memory_cos = self.get_each_batch_sample_sim().clone()#estimate a score for each added sample
                            self.sampled_memory_cos = torch.cat((self.sampled_memory_cos, this_sampled_memory_cos.clone()),
                                                                dim=0)
                            self.sampled_memory_data = torch.cat((self.sampled_memory_data ,self.memory_data[added_inds].clone()),dim=0)
                            self.sampled_memory_labs = torch.cat(( self.sampled_memory_labs,self.memory_labs[added_inds].clone()),dim=0) 
                            
                    self.mem_cnt = 0
                    self.train()
                    
                           
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
    return test_loss
