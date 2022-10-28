import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from config import ET_FREQ

# Functions in this file are extracted from work of Elmadjian et al. on https://github.com/elmadjian/OEMC


class CNN_LSTM(nn.Module):

    def __init__(self, input_size, output_size, kernel_size, dropout, 
                 features, lstm_layers, conv_filters=(32, 16, 8), 
                 bidirectional=False):
        super(CNN_LSTM, self).__init__()
        self.conv_filters = conv_filters
        self.lstm_layers = lstm_layers
        conv_layers = []
        padding = int(np.floor((kernel_size-1)/2))
        for i, filter in enumerate(self.conv_filters):
            input_conv = input_size if i == 0 else conv_filters[i-1]
            if i > 0:
                conv_layers += [nn.Dropout(dropout)]
            conv_layers += [nn.Conv1d(input_conv, conv_filters[i], kernel_size,
                                      padding=padding, padding_mode='replicate')]
            conv_layers += [nn.BatchNorm1d(conv_filters[i])]
            conv_layers += [nn.ReLU()]        
        self.conv_layers = nn.Sequential(*conv_layers)
        self.flatten = TimeDistributed(nn.Flatten(), batch_first=True)
        hidden_state = 32
        if bidirectional:
            hidden_state = 16
        self.lstm = nn.LSTM(input_size=features, bidirectional=bidirectional,
                            hidden_size=hidden_state, num_layers=lstm_layers, 
                            batch_first=True)
        linear = nn.Linear(32, output_size)
        linear.weight.data.normal_(0, 0.01)
        self.output = TimeDistributed(linear, batch_first=True)


    def forward(self, x):
        out = x
        for layer in self.conv_layers:
            out = layer(out)
        out = self.flatten(out)
        out,_ = self.lstm(out)
        out = self.output(out[:,-1,:])
        out = F.log_softmax(out, dim=1)
        return out


class TimeDistributed(nn.Module):

    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)

        if self.batch_first:
            y = y.contiguous().view(x.size(0),-1,y.size(-1))
        else:
            y = y.view(-1, x.size(1), y.size(-1))
        return y


def train(model, optimizer, x, y):
    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss = F.nll_loss(output, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, x, y):
    model.eval()
    with torch.no_grad():
        output = model(x)
        loss = F.nll_loss(output, y, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1]
        preds = pred.view(pred.numel())
        return preds, y, loss


def create_batches(X, Y, start, end, timesteps, randomize=False):
        '''
        Creates a minibatch composed of one or more timesteps that
        can be fed into the network
        '''
        # if timesteps == 1:
        #     return create_batches_single(X, Y, start, end)
        b_Y = Y[start-1:end-1]
        b_X = np.array([X[i-timesteps:i,:] for i in range(start, end)])
        if randomize:
            shuffler = np.random.permutation(len(b_Y))
            b_X = b_X[shuffler]
            b_Y = b_Y[shuffler]
        batch_X = torch.from_numpy(b_X).float()
        batch_Y = torch.from_numpy(b_Y).long()
        if torch.cuda.is_available():
            batch_X = batch_X.cuda()
            batch_Y = batch_Y.cuda()
        return batch_X, batch_Y




def execute(x, y):
    


    learning_rate = 0.01
    timesteps = 25
    n_classes = 4
    kernel_size = 5
    dropout = 0.25
    features = 18
    epochs = 25
    num_batches = 587
    batch_size = 2048
    rand = True
    train_size = len(x)

    model = CNN_LSTM(timesteps, n_classes, kernel_size, dropout,
                          features, lstm_layers=2, bidirectional=True)

    model.cuda()

    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    steps = 0
    for epoch in range(1, epochs+1):
        cost = 0
        for k in range(10):
            start, end = k * batch_size, (k+1) * batch_size
            if start == 0:
                start = timesteps
            trainX, trainY = create_batches(x, y, start, end, timesteps, rand) 
            cost += train(model, optimizer, trainX, trainY)
            steps += 1
            # if k > 0 and (k % (num_batches//10) == 0 or k == num_batches-1):
            if True:
                print('Train Epoch: {:2} [{}/{} ({:.0f}%)]  Loss: {:.5f}  Steps: {}'.format(
                    epoch, end, train_size,
                    100*k/num_batches, cost/num_batches, steps 
                ), end='\r')
            
        preds, labels, loss = test(model, trainX, trainY)



    for epoch in range(1, epochs+1):
        cost = 0
        cost += train(model, optimizer, x, y)

        preds, labels, loss = test(model, x, y)

        print("check now")