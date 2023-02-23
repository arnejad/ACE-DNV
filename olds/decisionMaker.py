import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sklearn
import numpy as np
from torch.autograd import Variable
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


def data_prep(x, y, ratio=0.20, onehot=False):

  # x, y = balance_multinomial(x,y)
  oversample = RandomOverSampler(sampling_strategy='all')
  x, y = oversample.fit_resample(x, y)


  y[y==5] = 4
  y = y-1

  trainFeat, testFeat, trainLbl, testLbl = sklearn.model_selection.train_test_split(x, y, test_size = ratio, random_state = 5)

  if onehot:
    b = np.zeros((trainLbl.size, 4))
    b[np.arange(trainLbl.size), trainLbl] = 1
    trainLbl = b
    b = np.zeros((testLbl.size, 4))
    b[np.arange(testLbl.size), testLbl] = 1
    testLbl = b

  trainFeat = torch.from_numpy(trainFeat).float()
  testFeat = torch.from_numpy(testFeat).float()
  trainLbl = torch.from_numpy(trainLbl)
  testLbl = torch.from_numpy(testLbl)

  return trainFeat, testFeat, trainLbl, testLbl


class CONV2D_MODEL(nn.Module):
    def __init__(self):
      super(CONV2D_MODEL, self).__init__()
      self.conv0 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
      self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)

      self.fc0 = nn.Linear(32, 64)
      self.fc1 = nn.Linear(64, 128)
      self.fc2 = nn.Linear(128, 128)
      self.fc3 = nn.Linear(128, 64)
      self.fc4 = nn.Linear(64, 32)
      self.fc5 = nn.Linear(32, 16)
      self.fc6 = nn.Linear(16, 8)
      self.fc7 = nn.Linear(8, 4)
      self.sm = nn.Softmax()

    def forward(self, x):
      
      out = self.conv0(x)
      out = self.conv1(out)
      out = torch.flatten(out)

      out = torch.relu(self.fc0(out))
      out = self.fc1(out)
      out = torch.relu(self.fc2(out))
      out = self.fc3(out)
      out = torch.relu(self.fc4(out))
      out = self.fc5(out)
      out = torch.relu(self.fc6(out))
      out = self.fc7(out).softmax(dim=0)

      return out

def train_nn2(x, y):
  
  is_cuda = torch.cuda.is_available()
  if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
  else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
    
  trainFeat = np.zeros((1,1,20,6), dtype=np.float32)
  for i in range(10, len(x)-10):
    trainFeat = np.append(trainFeat, [[x[i-10:i+10]]], axis=0)

  trainFeat = np.delete(trainFeat, 0,0) 
  trainFeat = torch.from_numpy(trainFeat).float()
  trainFeat = trainFeat.to(device)
  # trainFeat = np.array(trainFeat)
  y = y[10:-10]
  y[y==5] = 4
  y = y-1

  y = torch.from_numpy(y)
  y = y.to(device)

  model = CONV2D_MODEL().to(device)
  model.train()

  

  # output = model(trainFeat[0].to(device))

  loss_function = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

  total_losses = []
  losses = []
  # losses_valid = []
  for epoch in range(2):

        # zero the parameter gradients
      # optimizer.zero_grad()
      losses = []
      for i in range(len(trainFeat)):
        pred_y = model(trainFeat[i])
        pred_y = torch.squeeze(pred_y)
        # pred_y = torch.argmax(pred_y_pre, dim=1)
        loss = loss_function(pred_y, y[i])
        # loss =  torch.autograd.Variable(loss, requires_grad = True)

        losses.append(loss.item())

        loss.backward()

        optimizer.step()

        # pred_y_valid = model(testFeat)
        # loss_val = loss_function(pred_y_valid, testLbl)
        # loss_val =  torch.autograd.Variable(loss_val, requires_grad = True)
        # losses_valid.append(loss_val.item())
      
      total_losses.append(np.sum(losses))

      if epoch % 1 == 0:
          print("Epoch {} train loss: {} ".format(epoch, np.sum(losses)))
  
  #evaluate 
  model.eval()
  preds = []
  for i in range(len(trainFeat)):
    pred_y = model(trainFeat[i])
    pred_y = torch.argmax(pred_y)
    preds.append(pred_y.item())
    


   
  predicted = torch.max(pred_y, 1)
  acc = accuracy(preds, y)
  plt.plot(total_losses)
  # plt.plot(losses_valid)
  plt.ylabel('loss')
  plt.xlabel('epoch')
  # plt.title("Learning rate %f"%(learning_rate))
  plt.show()

  print("TRAINING ")
