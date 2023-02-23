import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sklearn
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from imblearn.over_sampling import RandomOverSampler 

is_cuda = torch.cuda.is_available()
if is_cuda:
  device = torch.device("cuda")
  print("GPU is available")
else:
  device = torch.device("cpu")
  print("GPU not available, CPU used")

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



class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True, nonlinearity='relu')  
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        # hidden = self.init_hidden(batch_size)
        h0 = Variable(torch.zeros(self.n_layers, x.size(0), self.hidden_dim))

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, h0)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        # out = self.fc(out[:, -1, :]) 
        
        return out
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden




class LSTM(nn.Module):

    def __init__(self, dimension=32):
        super(LSTM, self).__init__()

        # self.embedding = nn.Embedding(4, 300)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=6,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True)
        # self.drop = nn.Dropout(p=0.5)

        self.fc0 = nn.Linear(32, 4)
        self.fc1 = nn.Linear(16, 4)

    def forward(self, x):

        # text_emb = self.embedding(text)

        # packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(x)
        # output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out = output[0]
        # out_reverse = output[:, 0, self.dimension:]
        # out_reduced = torch.cat((out_forward, out_reverse), 1)
        # out = self.drop(out_forward)
        out = torch.relu(self.fc0(out))
        # out = torch.relu(self.fc1(out))
        # out = torch.squeeze(out, 1)
        out = torch.sigmoid(out)

        return out

# def sequenceMaker(x,y, l):
#   for 


def train_LSTM(x, y):
  
  is_cuda = torch.cuda.is_available()
  if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
  else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
  

  
  trainFeat, testFeat, trainLbl, testLbl = data_prep(x,y, ratio=0.20, onehot=True)

  model = LSTM().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  criterion = nn.CrossEntropyLoss()

  window_size = 20
  sideLen = int(window_size/2)
  # initialize running values
  eval_every = 1
  num_epochs = 5
  running_loss = 0.0
  valid_running_loss = 0.0
  global_step = 0
  train_loss_list = []
  valid_loss_list = []
  global_steps_list = []

  trainLbl = trainLbl.to(device)
  trainFeat = trainFeat.to(device)
  # training loop
  model.train()
  for epoch in range(num_epochs):
    running_loss = 0.0
    valid_running_loss = 0.0
    for i in range(sideLen, len(trainFeat)-sideLen):
        
        # window_size = window_size.to(device)

        output = model(trainFeat[i-sideLen:i+sideLen])
        # output = output[:, -1, :]
        # output = output.argmax(dim=0)
        loss = criterion(output, trainLbl[i].to(torch.float32))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
        # update running values
        running_loss += loss.item()
        global_step += 1
    
        # # evaluation step
        # if global_step % eval_every == 0:
        #     model.eval()
        #     with torch.no_grad():                    
        #       # validation loop
        #       for (labels, (title, title_len), (text, text_len), (titletext, titletext_len)), _ in valid_loader:
        #           labels = labels.to(device)
        #           titletext = titletext.to(device)
        #           titletext_len = titletext_len.to(device)
        #           output = model(titletext, titletext_len)

        #           loss = criterion(output, labels)
        #           valid_running_loss += loss.item()

        #     # evaluation
        #     average_train_loss = running_loss / eval_every
        #     average_valid_loss = valid_running_loss / len(valid_loader)
        #     train_loss_list.append(average_train_loss)
        #     valid_loss_list.append(average_valid_loss)
        #     global_steps_list.append(global_step)

        #     # resetting running values
        #     running_loss = 0.0                
        #     valid_running_loss = 0.0
        #     model.train()

        #     # print progress
        #     print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
        #           .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
        #                   average_train_loss, average_valid_loss))
            
            # # checkpoint
            # if best_valid_loss > average_valid_loss:
            #     best_valid_loss = average_valid_loss
            #     save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
            #     save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)

  print('Finished Training!')




def train_nn(x, y):

    
  trainFeat, testFeat, trainLbl, testLbl = data_prep(x,y, ratio=0.20)

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



def train_nn2(x, y):

  new_trainFeat = []
  for i in range(10, len(x)):
    new_trainFeat.append(x[i-10:i+10])

  trainFeat, testFeat, trainLbl, testLbl = data_prep(new_trainFeat,y[10:-10], ratio=0.20)

  

  trainFeat = trainFeat.to(device)

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


    
  trainFeat, testFeat, trainLbl, testLbl = data_prep(x,y, ratio=0.20)

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


def train_rnn(x, y):
  print("training RNN")

  

  # Instantiate the model with hyperparameters
  model = Model(input_size=6, output_size=4, hidden_dim=4, n_layers=1)
  # We'll also set the model to the device that we defined earlier (default is CPU)
  model.to(device)

  

  trainFeat, testFeat, trainLbl, testLbl = data_prep(x,y, ratio=0.20, onehot=True)

  n_epochs = 100
  lr=0.01
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    trainFeat.to(device)
    output = model(trainFeat)
    loss = criterion(output, trainLbl.view(-1).long())
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly
    
    if epoch%10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))