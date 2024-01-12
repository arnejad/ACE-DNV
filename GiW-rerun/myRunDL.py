## This code has been written by Ashkan Nejad to rerun the Gaze-in-Wild for comparison 
# to rerun gaze-in-wild using our scores, put this file in the same dir as runDL.py
# copy our replace the original opt.py with ours in this repo. 
import os
import torch
import pickle
import numpy as np
import scipy.io as scio
from DeepModels.args import parse_args
from DeepModels.DataLoader import GIW_readSeq
from DeepModels.opts import test
from DeepModels.models import *

args = parse_args()
args.prec = torch.float64
path2weights = '/media/ash/Expansion/GiW/Pretrained_models/RNN'
f = open(os.path.join(os.path.join('/media/ash/Expansion/GiW/code/ML/DeepModels/', 'Data'), 'Data.pkl'), 'rb')
seq = pickle.load(f)[1]
ID_info = np.stack(seq['id'], axis=0)

PrList = [1, 2, 3, 6, 8, 9, 12, 16, 17, 22]
ModelPresent = list(range(0, 9))
ModelPresent = [x for x in ModelPresent if x not in [5, 6, 8]] # Remove these from analysis
ModelID = [14, 24, 34, 44, 54, 64, 74, 84, 94]
All_S_e = list()
All_S_s = list()
All_Supp_e = list()
All_Supp_s = list()

for PrIdx in PrList:
    print('Evaluating PrIdx: {}'.format(PrIdx))
    testObj = GIW_readSeq(seq, PrIdx)
    testloader = torch.utils.data.DataLoader(testObj,
                                            batch_size=1,
                                            num_workers=1,
                                            shuffle=False)

    for model_num in ModelPresent:
        print('eval model: {}'.format(model_num+1))
        model = eval('model_{}'.format(model_num+1))
        net = model().cuda().to(args.prec)
        best = 0
        f1s_sample = []
        f1s_event = []
        supp_sample = []
        supp_event = []
        for fold in range(0, args.folds):
            print('fold: {}'.format(fold))
            path2weight = os.path.join(path2weights, 'PrTest_{}_model_{}_fold_{}.pt'.format(int(PrIdx), int(model_num+1), fold))
            if os.path.exists(path2weight):
                try:
                    net.load_state_dict(torch.load(path2weight)['net_params'])
                except:
                    print('Dict mismatch. Training not complete yet.')
                    continue
                f1_ie, f1_is, supp_e, supp_s = test(net, testloader, args, talk=False)

                f1s_sample.append(f1_is)
                f1s_event.append(f1_ie)
                supp_sample.append(supp_s)
                supp_event.append(supp_e)
            else:
                print('Weights for this model does not exist')

        with open('/media/ash/Expansion/GiW/code/ML/log.txt', 'a') as file:
            file.write("model num: " + str(model_num) + " participant: " + str(PrIdx) + "\n")

    f1_samp_avg = np.mean(f1s_sample, axis=0)
    f1_event_avg = np.mean(f1s_event, axis=0)

    All_S_e.append(f1_event_avg)
    All_S_s.append(f1_samp_avg)
    All_Supp_e.append(supp_e)
    All_Supp_s.append(supp_s)


    f1_samp_avg = np.sum(f1s_sample*(supp_sample/np.sum(supp_sample, axis=0)), axis=0)
    f1_event_avg =  np.sum(f1s_event*(supp_event/np.sum(supp_event, axis=0)), axis=0)

All_S_s = np.array(All_S_s)
All_S_e = np.array(All_S_e)

np.savetxt("/media/ash/Expansion/GiW/All_S_e.csv", All_S_e, delimiter=",")
np.savetxt("/media/ash/Expansion/GiW/All_S_s.csv", All_S_s, delimiter=",")


