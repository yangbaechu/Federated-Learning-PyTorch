from __future__ import print_function
import torch
import torch.nn as nn
import pandas as pd
# import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import numpy as np
import early_stopping as es
# import tensorboard

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class MyDataset(torch.utils.data.Dataset):
    """ dataset."""

    # Initialize your data, download, etc.
    def __init__(self, df_X, df_Y):
        self.len = df_X.shape[0]
        self.x_data = torch.tensor(df_X.values, dtype = torch.float32)
        self.y_data = torch.tensor(df_Y.values, dtype = torch.float32)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_sizes, num_output):
            # the number of hidden layer >= 1
            super(NeuralNet, self).__init__()
            nhiddens = len(hidden_sizes)
            self.fcs = []
            self.fcs.append(nn.Linear(input_size, hidden_sizes[0]).float())  # input - hidden
            for i in range(nhiddens - 1):   # middle
                self.fcs.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]).float())
            self.fcs.append(nn.Linear(hidden_sizes[-1], num_output).float()) # output hidden
            self.relu = nn.ReLU()

            for i in range(nhiddens + 1):   # the number of hidden fcs (nhiddens) + in/out fcs (2)
                self.add_module("fc" + str(i), self.fcs[i])
            
            #self.float()
        
        def forward(self, x):
            out = x
            for fc in self.fcs:
                out = self.relu(fc(out))
            return out

def get_train_valid_test(df_X, df_Y, ratio):
    # data split, 주어진비율대로 data를 나눈다
    ratio = ratio / ratio.sum()
    new = ratio[2]/(ratio[1]+ratio[2])
    x_train, x_valid, y_train, y_valid = train_test_split(df_X, df_Y, test_size = ratio[1]+ratio[2] , shuffle = True)    # train - valid
    x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size = new , shuffle = True)    # train - valid
    return x_train, x_valid, x_test, y_train, y_valid, y_test

def print_dic(model, optimizer):
    # 모델의 state_dict 출력
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # 옵티마이저의 state_dict 출력
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

def test_model(model, test_loader, criterion, device):

    # test loss 및 accuracy을 모니터링하기 위해 list 초기화
    test_loss = 0.0

    # 모델이 학습되는 동안 test  loss를 track
    test_losses = []
    # epoch당 average test loss를 track

    predicted_spectral_list = []
    model.eval() # prep model for evaluation

    x_info = []
    for x, y in test_loader:
        
        x, y = x.to(device), y.to(device)
        
        # forward pass: 입력을 모델로 전달하여 예측된 출력 계산
        y_predicted = model(x)

        # calculate the loss
        MSEloss = criterion(y_predicted, y)
        
        # update test loss
        # test_loss += loss.item()*x.size(0)
        test_losses.append(MSEloss.item())
        predicted_spectral_list.append(y_predicted[0].detach().cpu().numpy())

        x_info.append(x[0].detach().cpu().numpy())
    print(f"TEST RESULT: average MSEloss = {np.average(test_losses)} ")

    return test_losses, predicted_spectral_list, x_info

# def do_learning(df_X, df_Y, resume = False):
def do_learning(model, optimizer, scheduler, device, start_epoch, num_epochs, batch_size, patience, avg_train_losses, avg_valid_losses,
train_loader, valid_loader, criterion,  train_idxs, valid_idxs, test_idxs, adjust_learning_rate = False):

    early_stopping = es.EarlyStopping(patience=patience, verbose=True)

    # 모델이 학습되는 동안 trainning loss를 track
    train_losses = []
    # 모델이 학습되는 동안 validation loss를 track
    valid_losses = []

    # Train the model
    total_step = len(train_loader)
    for epoch in range(start_epoch, num_epochs):

        ###################
        # train the model #
        ###################
        model.train()
        for x, y in train_loader:
            # Forward pass
            x, y = x.to(device), y.to(device)
            Y_predicted = model(x)
            loss = criterion(Y_predicted, y)
            
            # Backward
            loss.backward()

            # Update
            optimizer.step()    
            optimizer.zero_grad()

            # record training loss
            train_losses.append(loss.item())
        
        # if (epoch+1) % lr_epoch_interval == 0:
        if adjust_learning_rate:
            scheduler.step()
            print(f'[{epoch}/{num_epochs}] learning rate = {get_lr(optimizer)}')

        ######################    
        # validate the model #
        ######################
        model.eval()
        # for x, y in valid_loader:
        for x, y in valid_loader:

            x, y = x.to(device), y.to(device)
            # forward pass: 입력된 값을 모델로 전달하여 예측 출력 계산
            Y_predicted = model(x)
            # calculate the loss
            loss = criterion(Y_predicted, y)
            # record validation loss
            valid_losses.append(loss.item())

        # print 학습/검증 statistics
        # epoch당 평균 loss 계산
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        print(f'[{epoch}/{num_epochs}] '
        + f'train_loss: {train_loss:.5f}' 
        + f' valid_loss: {valid_loss:.5f}')

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping는 validation loss가 감소하였는지 확인이 필요하며,
        # 만약 감소하였을경우 현재 모델을 checkpoint로 만든다.
        # early_stopping(valid_loss, model)
        early_stopping(valid_loss, model, optimizer, scheduler, epoch, batch_size,
         avg_train_losses, avg_valid_losses, train_idxs, valid_idxs, test_idxs)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    ######################
      # TEST the model #
    ######################
    # test_losses, predicted_spectral_list = test_model(model, test_loader, criterion, device)


    # df_result_prediction = pd.DataFrame(predicted_spectral_list)
    # df_result_test_losses = pd.DataFrame(test_losses, columns=['loss'])
    # df_result_test_losses.index = df_result_prediction.index =  x_test.index

    # best model이 저장되어있는 last checkpoint를 로드한다.
    # model.load_state_dict(checkpoint['model_state_dict'])
    # return  model, avg_train_losses, avg_valid_losses, df_result_test_losses, df_result_prediction, x_train.index, x_valid.index
    return  model, avg_train_losses, avg_valid_losses, early_stopping
