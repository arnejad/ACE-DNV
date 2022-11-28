import numpy as np
from modules.PatchSimNet import perdict
from modules.wildMove import VOAnalyzer
import cv2 as cv
import modules.visualizer as visual
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from modules.decisionMaker import print_scores as print_scores
from sklearn.model_selection import train_test_split
import torch

from config import PATCH_SIM_THRESH, GAZE_DIST_THRESH, ENV_CHANGE_THRESH, PATCH_SIZE, LAMBDA, PATCH_PRIOR_STEPS, OUT_DIR

def eventDetector_new(feats, lbls):


    feats = np.squeeze(feats)
    feats = np.concatenate(feats)

    lbls = np.squeeze(lbls)
    lbls = np.concatenate(lbls)

    rmInd = np.where(lbls==3)[0]

    # lbls[rmInd] = 1
    lbls = np.delete(lbls, rmInd[:8000])
    feats = np.delete(feats, rmInd[:8000], axis=0)
    # indices = np.random.permutation(np.random.rand(data_len, 1))
    # divider_idx = int(test_ratio*data_len)
    # training_idx, test_idx = indices[divider_idx:], indices[:divider_idx]

    X_train, X_test, y_train, y_test = train_test_split(feats, lbls, test_size=0.33, random_state=42)


    clf = RandomForestClassifier(random_state=0, criterion='gini', n_estimators=40)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    preds = torch.from_numpy(preds)
    lbls = torch.from_numpy(y_test)
    print_scores(preds, lbls, 0, 'RF')

    

    lbl_list = np.unique(lbls)
    for c in lbl_list:
        y = np.array(y_train)
        y[y!=c] = 10
        y[y==c] = 1
        y[y==10] = 0
        clf = RandomForestClassifier(random_state=0, criterion='gini', n_estimators=40)
        clf.fit(X_train, y)
        preds = clf.predict(X_test)
        preds = torch.from_numpy(preds)
        y = np.array(y_test)
        y[y!=c] = 10
        y[y==c] = 1
        y[y==10] = 0
        lbls = torch.from_numpy(y)
        print_scores(preds, lbls, 0, 'RF')













    
    fig, axs = visual.knowledgePanel_init()

    feats = feats[0]
    lbls = lbls[0]


    patchSims = feats[:, 4]
    gazeVels = feats[:, 0]
    gazeDirs = feats[:,1]
    headVels = feats[:,2]
    headDirs = feats[:,3]

    PATCH_SIM_THRESH = 75
    ENV_CHANGE_THRESH = 0.01
    GAZE_DIST_THRESH = 4
    GAZE_SMOOTHSLIDE_THRESH = 0.07
    decisions = []
    for i in range(len(feats)):

        patchSim = patchSims[i]
        gazeVel = gazeVels[i]
        gazeDir = gazeDirs[i]
        headVel = headVels[i]
        headDir = headDirs[i]
        
        decision = 0
        # final decision
        visual.knowledgePanel_update(axs, None, np.column_stack((patchSims[:i+1], gazeVels[:i+1], headVels[:i+1], lbls[:i+1])))
        plt.pause(0.0000001)
        
        if patchSim < PATCH_SIM_THRESH:
            if gazeVel > GAZE_DIST_THRESH:
                decision = 2 #saccade
            elif gazeVel < GAZE_DIST_THRESH:
                if headVel < ENV_CHANGE_THRESH:
                    decision = 0 #fixation
        
        else:# patchDist < GAZE_SIM_THRESH:
            if gazeVel > GAZE_SMOOTHSLIDE_THRESH:
                if headVel > ENV_CHANGE_THRESH:
                    decision = 3 #"gaze following" or tVOR and optokinetic
                else:
                    decision = 1 #"Gaze Pursuit"
            else:
                # if envMag > ENV_CHANGE_THRESH:
                #     decision = 4 #"HeadPursuit"
                # else:
                    decision = 1 #"fixation"

        if (decision == lbls[i]):
            print("correct {} = {}".format(decision, lbls[i]))
        else:
            print("Wrong: {} instead of {}".format(decision, lbls[i]))
        decisions.append(decision)


def  eventDetector(patchSim, gazeDists, orientChange, lbls):
    
    # patchDist = patchContent.compare_old(patch1, patch2) #compute the patch content similarity
    

    # atten_flow = OFAnalyzer(frameNum, gazeCoord1, gazeCoord2)
    # envMag = VOAnalyzer(frameNum)

    # print(magMean)
    
    fig, axs = visual.knowledgePanel_init()    #initiallization of knowledge panel


    PATCH_SIM_THRESH = 75
    ENV_CHANGE_THRESH = 5.1
    GAZE_DIST_THRESH = 4
    GAZE_SMOOTHSLIDE_THRESH = 100
    decisions = []
    for i in range(len(gazeDists)):

        
        patchSimAvg = patchSim[i]
        gazeDist = gazeDists[i]
        envMag = orientChange[i]
        
        decision = 0
        # final decision
        visual.knowledgePanel_update(axs, None, np.column_stack((patchSim[:i+1], gazeDists[:i+1], orientChange[:i+1])))
        plt.pause(0.0000001)
        
        if patchSimAvg < PATCH_SIM_THRESH:
            if gazeDist > GAZE_DIST_THRESH:
                decision = 3 #saccade
            elif gazeDist < GAZE_DIST_THRESH:
                if envMag < ENV_CHANGE_THRESH:
                    decision = 1 #fixation
        
        else:# patchDist < GAZE_SIM_THRESH:
            if gazeDist > GAZE_SMOOTHSLIDE_THRESH:
                if envMag > ENV_CHANGE_THRESH:
                    decision = 5 #"fixWithHeadMove" or tVOR and optokinetic
                else:
                    decision = 2 #"SmoothPursuit"
            else:
                # if envMag > ENV_CHANGE_THRESH:
                #     decision = 4 #"HeadPursuit"
                # else:
                    decision = 1 #"fixation"

        if (decision == lbls[i]):
             print("correct {} = {}".format(decision, lbls[i]))
        else:
             print("Wrong: {} instead of {}".format(decision, lbls[i]))
        decisions.append(decision)
             


        ############ OLD VERSION

        # if patchSimAvg < PATCH_SIM_THRESH:
        #     if gazeDist > GAZE_DIST_THRESH:
        #         decision = "PotSAC"
        #     elif gazeDist < GAZE_DIST_THRESH:
        #         if envMag < ENV_CHANGE_THRESH:
        #             decision = "fixation"
        
        # else:# patchDist < GAZE_SIM_THRESH:
        #     if gazeDist > GAZE_DIST_THRESH:
        #         if envMag > ENV_CHANGE_THRESH:
        #             decision = "fixWithHeadMove"
        #         else:
        #             decision = "SmoothPursuit"
        #     else:
        #         if envMag > ENV_CHANGE_THRESH:
        #             decision = "HeadPursuit"
        #         else:
        #             decision = "fixation"

        # if decision == "":
        #     decision = "None"


    # return decision, [[patchDist.item(), patchSimAvg, gazeDist, envMag]]
    # fig.show()

    return decisions

    
