from requests import patch
import numpy as np
from modules.PatchSimNet import perdict
from modules.wildMove import VOAnalyzer
import cv2 as cv
import modules.visualizer as visual
import matplotlib.pyplot as plt


from config import PATCH_SIM_THRESH, GAZE_DIST_THRESH, ENV_CHANGE_THRESH, PATCH_SIZE, LAMBDA, PATCH_PRIOR_STEPS


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

    
