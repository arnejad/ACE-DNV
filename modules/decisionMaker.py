import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import random
from sklearn.utils import class_weight
from sklearn import utils



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
    # loss = F.nll_loss(output, y, weight= torch.tensor([1,0,1,0.15]).cuda(),reduction='mean')
    loss = F.nll_loss(output, y,reduction='mean')
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, x, y):
    model.eval()
    with torch.no_grad():
        output = model(x)
        loss = F.nll_loss(output, y, reduction='sum', ).item()
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


def batchMaker(X, Y, batch_size, timesteps, randomize=False):
    timeseries = []
    all_batches_X = []
    all_batches_Y = []
    
    for db in range(len(X)):
        B = int(len(X[db])/batch_size) # number of batches to be extracted from each dataframe
        for b in range(1, B-1):
            start, end = b * batch_size, (b+1) * batch_size
            b_Y = Y[db][start-1:end-1]

            b_X = np.array([X[db][i-timesteps:i,:] for i in range(start, end)])
            # timeseries.append(b_X)
            if randomize:
                shuffler = np.random.permutation(len(b_Y))
                b_X = b_X[shuffler]
                b_Y = b_Y[shuffler]
            batch_X = torch.from_numpy(b_X).float()
            batch_Y = torch.from_numpy(b_Y).long()
            if torch.cuda.is_available():
                batch_X = batch_X.cuda()
                batch_Y = batch_Y.cuda()
            all_batches_X.append(batch_X)
            all_batches_Y.append(batch_Y)

    return all_batches_X, all_batches_Y

def batchMaker_new(X, Y, batch_size, timesteps, train):
    # this batchmaker shuffles all the data (not innter-batch) and has a window not for [i-timestep, i]
    # but [i-(timestep/2), i+(timestep/2)]

    timeseries = []
    lbls_all = []
    
    # X = np.array(X, dtype=object); Y = np.array(Y, dtype=object)
    for db in range(len(X)):
        
        serie_X = np.array([X[db][i-int(timesteps/2):i+(int(timesteps/2)+1),:] for i in range(int(timesteps/2),len(X[db])-int((timesteps/2)))])
        lbls = Y[db][int(timesteps/2):len(X[db])-int(timesteps/2)]
        timeseries.append(serie_X)
        lbls_all.append(lbls)

    timeseries = np.squeeze(np.array(timeseries, dtype=object)) 
    Y = np.squeeze(np.array(lbls_all, dtype=object))

    # #temp delete gaze followings
    # rmInd = np.where(Y==3)[0]

    # Y = np.delete(Y, rmInd[6000:])
    # timeseries = np.delete(timeseries, rmInd[6000:], axis=0)

    if train:
        timeseries = np.concatenate(timeseries); Y = np.concatenate(Y)

    timeseries, Y = utils.shuffle(timeseries, Y)


    all_batches_X = []
    all_batches_Y = []

    B = int(len(timeseries)/batch_size) # number of batches to be extracted from each dataframe
    for b in range(1, B-1):
        start, end = b * batch_size, (b+1) * batch_size
        b_X = timeseries[start-1:end-1]
        b_Y = Y[start-1:end-1]

        b_X = np.array(b_X, dtype=np.float64)
        b_Y = np.array(b_Y, dtype=np.int8)
        batch_X = torch.from_numpy(b_X).float()
        batch_Y = torch.from_numpy(b_Y).long()
        if torch.cuda.is_available():
            batch_X = batch_X.cuda()
            batch_Y = batch_Y.cuda()
        all_batches_X.append(batch_X)
        all_batches_Y.append(batch_Y)

    return all_batches_X, all_batches_Y

def f1_score(preds, labels, class_id):
    '''
    preds: precictions made by the network
    labels: list of expected targets
    class_id: corresponding id of the class
    '''
    true_count = torch.eq(labels, class_id).sum()
    true_positive = torch.logical_and(torch.eq(labels, preds),
                                      torch.eq(labels, class_id)).sum().float()
    precision = torch.div(true_positive, torch.eq(preds, class_id).sum().float())
    precision = torch.where(torch.isnan(precision),
                            torch.zeros_like(precision).type_as(true_positive),
                            precision)
    recall = torch.div(true_positive, true_count)
    f1 = 2*precision*recall / (precision+recall)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1).type_as(true_positive),f1)
    return f1.item()


def print_scores(total_pred, total_label, test_loss, name):
    f1_fix = f1_score(total_pred, total_label, 0)*100
    f1_gazeP = f1_score(total_pred, total_label, 1)*100
    f1_sacc = f1_score(total_pred, total_label, 2)*100
    f1_gazeF = f1_score(total_pred, total_label, 3)*100
    f1_avg = (f1_fix + f1_sacc + f1_gazeP + f1_gazeF)/4
    print('\n{} set: Average loss: {:.4f}, F1_FIX: {:.2f}%, F1_SACC: {:.2f}%, F1_GazePursuit: {:.2f}%, F1_GazeFollowing: {:.2f}%, AVG: {:.2f}%\n'.format(
        name, test_loss, f1_fix, f1_sacc, f1_gazeP, f1_gazeF, f1_avg
    ))
    return (f1_fix + f1_sacc + f1_gazeP + f1_gazeF)/4


def execute(x, y):
    


    learning_rate = 0.0001
    timesteps = 25
    n_classes = 2
    kernel_size = 5
    dropout = 0.25
    features = 6
    epochs = 3000
    num_batches = 587
    batch_size = 1024
    rand = True


    # classW = class_weight.compute_class_weight(class_weight='balanced', classes = [0,2,3], y = np.hstack(y))
    
    model = CNN_LSTM(timesteps, n_classes, kernel_size, dropout,
                          features, lstm_layers=2, bidirectional=True)

    model.cuda()


    # valid_x, valid_y = x[0], y[0]
    # x, y = np.delete(x,0), np.delete(y,0)

    # valid_batches_x, valid_batches_y = batchMaker_new([valid_x], [valid_y], batch_size, timesteps, train=False)

    allBatches_x, allBatches_y = batchMaker_new(x, y, batch_size, timesteps, train=True)
    valid_batches_x, valid_batches_y = allBatches_x[0], allBatches_y[0]
    trainBatches_X, trainBatches_Y = allBatches_x[1:], allBatches_y[1:]
    
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    scores = []
    steps = 0
    for epoch in range(1, epochs+1):
        cost = 0
        
        # trainBatches_X, trainBatches_Y = batchMaker_new(x,y, batch_size, timesteps, train=True)

        for k in range(len(trainBatches_Y)):
            cost += train(model, optimizer, trainBatches_X[k], trainBatches_Y[k])
            steps += 1
            # if k > 0 and (k % (num_batches//10) == 0 or k == num_batches-1):
            if True:
                print('Train Epoch: {:2} [({:.0f}%)]  Loss: {:.5f}  Steps: {}'.format(
                    epoch,
                    100*k/num_batches, cost/num_batches, steps 
                ), end='\r')
        
        # choose a random batch for validation
        
        
        # preds, labels, t_loss = test(model, trainBatches_X[k], trainBatches_Y[k])
        # k = random.randint(0, len(valid_batches_y)-1)
        preds, labels, t_loss = test(model, valid_batches_x, valid_batches_y)
        score = print_scores(preds, labels, t_loss, 'Val.')
        scores.append(score)

