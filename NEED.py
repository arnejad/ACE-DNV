from unittest.mock import patch
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from modules.decisionMaker import train_nn as trainDm_nn

import random

def balance (x, y):
    zerIndcs = np.where(y==0)
    oneIndcs = np.where(y==1)
    oneIndcs = oneIndcs[0]
    numOnes = len(oneIndcs)
    ix = np.random.choice(len(zerIndcs[0]), size=numOnes, replace=True)
    
    chosenZeros = zerIndcs[0][ix]
    np.random.shuffle(chosenZeros)

    feats = np.concatenate((x[chosenZeros, :], x[oneIndcs,:]))
    lbls = np.concatenate((np.zeros((1,numOnes)), np.ones((1,numOnes))), axis=1)
    lbls = np.transpose(lbls)
    return feats, lbls



def zScore_norm(featSet):
    #normalize data (z-score) 
    means = np.mean(featSet, axis=0)
    std = np.std(featSet, axis=0)
    featSet = (featSet-means)/std
    mins = np.min(featSet, axis=0)
    featSet = featSet+ np.abs(mins)
    return featSet

def train_logreg(featSet, lblSet):
    print("training initiated")
    

    #normalize data (z-score) 
    featSet = zScore_norm(featSet)


    lblSet[lblSet != 1] = 0
    lblSet[lblSet == 1] = 1

    featSet, lblSet = balance(featSet, lblSet)

    trainFeat, testFeat, trainLbl, testLbl = sklearn.model_selection.train_test_split(featSet, lblSet, test_size = 0.20, random_state = 5)

    # model1 = LogisticRegression(random_state=0, multi_class='multinomial', penalty='none', solver='newton-cg').fit(trainFeat, trainLbl)

    # fixation classifier


    model1 = LogisticRegression(random_state=0, solver='newton-cg').fit(trainFeat, trainLbl)

    preds = model1.predict(trainFeat)
    matches = len(np.where(preds == np.transpose(trainLbl)[0])[0])
    print(matches/len(trainLbl))


    preds = model1.predict(testFeat)
    matches = len(np.where(preds == np.transpose(testLbl)[0])[0])
    print(matches/len(testLbl))

    # lblsNotFive = testLbl[np.where(testLbl != 1)]
    # predsNotFive = preds[np.where(testLbl != 1)]
    # matches = len(np.where(predsNotFive == lblsNotFive)[0])
    # print(matches/len(lblsNotFive))

    return preds, testLbl

def train_nn(featSet, lblSet):

    print("training initiated")

    featSet = zScore_norm(featSet)
    trainDm_nn(featSet, lblSet)




