import os
import sys
import argparse
from functools import partial
from tqdm import tqdm
import numpy as np
import torch
import torchfile
from torchnet.dataset import ListDataset, ConcatDataset
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn import metrics
from scipy import interpolate
from torch.backends import cudnn
from urllib3 import Retry

CUDA_ID = '0'
model = '2ch2stream'
lua_model = '/home/ashdev/projects/Zagoruyko/models/2ch2stream/2ch2stream_notredame.t7'


def conv2d(input, params, base, stride=1, padding=0):
    return F.conv2d(input, params[base + '.weight'], params[base + '.bias'],
                    stride, padding)


def linear(input, params, base):
    return F.linear(input, params[base + '.weight'], params[base + '.bias'])



#####################   2ch2stream   #####################

def deepcompare_2ch2stream(input, params):

    def stream(input, name):
        o = conv2d(input, params, name + '.conv0')
        o = F.max_pool2d(F.relu(o), 2, 2)
        o = conv2d(o, params, name + '.conv1')
        o = F.max_pool2d(F.relu(o), 2, 2)
        o = conv2d(o, params, name + '.conv2')
        o = F.relu(o)
        o = conv2d(o, params, name + '.conv3')
        o = F.relu(o)
        # return o.view(o.size(0), -1)
        return torch.flatten(o)

    o_fovea = stream(F.avg_pool2d(input, 2, 2), 'fovea')
    o_retina = stream(F.pad(input, (-16,) * 4), 'retina')
    o = linear(torch.cat([o_fovea, o_retina], dim=0), params, 'fc0')
    return linear(F.relu(o), params, 'fc1')


model = deepcompare_2ch2stream

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_network():
    def cast(t):
        return torch.from_numpy(t).float().to(device) if torch.cuda.is_available() else t

    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_ID
    if torch.cuda.is_available():
        # to prevent opencv from initializing CUDA in workers
        torch.randn(8).cuda()
        os.environ['CUDA_VISIBLE_DEVICES'] = ''


    net = torchfile.load(lua_model)
    params = {}
    for j, branch in enumerate(['fovea', 'retina']):
        counter = 0
        for k, layer in enumerate(net.modules[0].modules[j].modules[1].modules):
            if layer.weight is not None:
                if k in [1,4,7,9]:
                    params['%s.conv%d.weight' % (branch, counter)] = layer.weight
                    params['%s.conv%d.bias' % (branch, counter)] = layer.bias
                    counter = counter+1
    
    counter = 0
    for k, layer in enumerate(net.modules):
        if layer.weight is not None:
            if k in [1,3]:
                params['fc%d.weight' % counter] = layer.weight
                params['fc%d.bias' % counter] = layer.bias
                counter = counter+1
        

    params = {k: Variable(cast(v)) for k, v in params.items()}
    
    return params


def perdict(input, params):
    

    y = model(torch.from_numpy(input).float().to(device), params)

    return y


