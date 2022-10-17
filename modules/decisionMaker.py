import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sklearn
import numpy as np

from imblearn.over_sampling import RandomOverSampler 

def balance_multinomial(x,y):
    reps,_ = np.histogram(y, bins=[1,2,3,4,5,6])
    maxim = np.max(reps)
    for c in np.unique(y):
        print("test")

def accuracy(preds, lbls):
  # assumes model.eval()
  # granular but slow approach
  n_correct = 0; n_wrong = 0
  for i in range(len(lbls)):

    if preds[i] == lbls[i]:
      n_correct += 1
    else:
      n_wrong += 1

  acc = (n_correct * 1.0) / (n_correct + n_wrong)
  return acc



def train_nn(x, y):

    # x, y = balance_multinomial(x,y)
    oversample = RandomOverSampler(sampling_strategy='all')
    x, y = oversample.fit_resample(x, y)


    y[y==5] = 4
    y = y-1

    trainFeat, testFeat, trainLbl, testLbl = sklearn.model_selection.train_test_split(x, y, test_size = 0.20, random_state = 5)
    trainFeat = torch.from_numpy(trainFeat).float()
    testFeat = torch.from_numpy(testFeat).float()
    trainLbl = torch.from_numpy(trainLbl)
    testLbl = torch.from_numpy(testLbl)

    learning_rate = 0.0001


    model = nn.Sequential(nn.Linear(11, 16),
                      nn.ReLU(),
                      nn.Linear(16, 32),
                      nn.ReLU(),
                      nn.Linear(32, 16),
                      nn.ReLU(),
                      nn.Linear(16, 8),
                      nn.ReLU(),
                      nn.Linear(8, 4))
                    #   nn.Sigmoid())
                      
    print(model)


    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


    losses = []
    losses_valid = []
    for epoch in range(100000):

         # zero the parameter gradients
        optimizer.zero_grad()

        pred_y = model(trainFeat)
        # pred_y = torch.argmax(pred_y_pre, dim=1)
        loss = loss_function(pred_y, trainLbl)
        # loss =  torch.autograd.Variable(loss, requires_grad = True)
        losses.append(loss.item())

        loss.backward()

        optimizer.step()

        pred_y_valid = model(testFeat)
        loss_val = loss_function(pred_y_valid, testLbl)
        # loss_val =  torch.autograd.Variable(loss_val, requires_grad = True)
        losses_valid.append(loss_val.item())
        if epoch % 100 == 0:
            print("Epoch {} train loss: {} , valid loss: {}".format(epoch, loss, loss_val))

    plt.plot(losses)
    plt.plot(losses_valid)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("Learning rate %f"%(learning_rate))
    plt.show()

    # acc = accuracy(pred_y_valid, testLbl)


    predicted = torch.max(pred_y_valid, 1)
    acc_valid = accuracy(predicted.indices, testLbl)

    predicted = torch.max(pred_y, 1)
    acc_train = accuracy(predicted.indices, trainLbl)

    # valid_acc = torch.sum(predicted == testLbl)/testLbl.shape[0]
    # predicted = torch.max(pred_y, 1)
    # train_acc = torch.sum(predicted == trainLbl)/testLbl.shape[0]
    
    print("train accuracy: {} validation accuracy: {}".format(acc_train, acc_valid))