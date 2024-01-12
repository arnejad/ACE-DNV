import numpy as np
# from modules.PatchSimNet import perdict
# from modules.wildMove import VOAnalyzer
# import cv2 as cv
import modules.visualizer as visual
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from modules.decisionMaker import print_scores as print_scores
from sklearn.model_selection import train_test_split
from modules.preprocess import data_stats
from modules.scorer import score
from modules.visualizer import showFeatureHist
import matplotlib.pyplot as plt
import torch
import pickle

from config import OUT_DIR, VISUALIZE

def suffle_trainAndTest(feats, lbls):


    feats = np.squeeze(feats)
    feats = np.concatenate(feats)

    lbls = np.squeeze(lbls)
    lbls = np.concatenate(lbls)

    #sample-level random split
    X_train, X_test, y_train, y_test = train_test_split(feats, lbls, test_size=0.2, random_state=42)
    
    # X_train, y_train, X_test, y_test = divider(feats,lbls)
    # X_train, y_train, y_train, y_test = train_test_split(X_train, y_train, test_size=1, random_state=42)

    
    # data_stats(lbls)

    clf = RandomForestClassifier(random_state=0, criterion='gini', n_estimators=300, max_features= 'log2', min_samples_leaf=1, max_depth=50, min_samples_split=2, bootstrap=False)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    pickle.dump(clf, open(OUT_DIR+'/models/RF.sav', 'wb'))
   
    f1_e, f1_s = score(preds, y_test)

    return f1_e, f1_s

    preds = torch.from_numpy(preds)
    lbls = torch.from_numpy(y_test)    
    print_scores(preds, lbls, 0, 'RF')

    pickle.dump(clf, open(OUT_DIR+'/models/RF.sav', 'wb'))

    

  


def pred_detector(feats, lbls, modelDir):

    # if len([feats.size]) > 1:
    feats = np.concatenate(feats)
    feats = np.squeeze(feats)
    
   
    lbls = np.squeeze(lbls)
    lbls = np.concatenate(lbls)


    clf = pickle.load(open(modelDir, 'rb'))
    preds = clf.predict(feats)

    
    # if lbls: 
    preds = torch.from_numpy(preds)
    lbls = torch.from_numpy(lbls)
        

    # if lbls: 
    # print_scores(preds, lbls, 0, 'RF')
    score(preds, lbls)

    return preds

    
def trainAndTest(x_train, y_train, x_test, y_test, validMethod):
    x_train = np.squeeze(x_train)
    x_train = np.concatenate(x_train)

    y_train = np.squeeze(y_train)
    y_train = np.concatenate(y_train)

    # x_train = x_train[:, 0:1]
    # 
    x_train, _ , y_train, _ = train_test_split(x_train, y_train, test_size=1, random_state=42) # just for shuffling

    if VISUALIZE: data_stats(y_train)

    x_test = np.squeeze(x_test)
    if validMethod=="TTS": x_test = np.concatenate(x_test)

    y_test = np.squeeze(y_test)
    if validMethod=="TTS": y_test = np.concatenate(y_test)

    # x_test, _ , y_test, _ = train_test_split(x_test, y_test, test_size=1, random_state=42) # just for shuffling

    # x_test = x_test[:, 0:1]

    # clf = RandomForestClassifier(random_state=0, criterion='gini', n_estimators=100, max_features= 'log2', min_samples_leaf=1, max_depth=10, min_samples_split=2, bootstrap=False)
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)

    f1_e, f1_s, supp_e, supp_s = score(preds, y_test)

    if VISUALIZE: showFeatureHist(clf)

    # for i in range(len(preds)):
    #     print("gt: " + str(y_test[i]) +" pred: " + str(preds[i]))

    return f1_e, f1_s, supp_e, supp_s


