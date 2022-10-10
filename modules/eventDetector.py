from requests import patch
import numpy as np
from modules.PatchSimNet import perdict
from modules.wildMove import VOAnalyzer
import cv2 as cv

from config import PATCH_SIM_THRESH, GAZE_DIST_THRESH, ENV_CHANGE_THRESH, PATCH_SIZE, LAMBDA, PATCH_PRIOR_STEPS


def  eventDetector(patch1, patch2, pastPatchDist, envMag, gazeCoord1, gazeCoord2, patchSimNet_params, frameNum):
    
    # patchDist = patchContent.compare_old(patch1, patch2) #compute the patch content similarity
    

    # atten_flow = OFAnalyzer(frameNum, gazeCoord1, gazeCoord2)
    envMag = VOAnalyzer(frameNum)

    # print(magMean)
    decision = ""
    # final decision
    if patchDistAvg < PATCH_SIM_THRESH:
        if gazeDist > GAZE_DIST_THRESH:
            decision = "PotSAC"
        elif gazeDist < GAZE_DIST_THRESH:
            if envMag < ENV_CHANGE_THRESH:
                decision = "fixation"
    
    else:# patchDist < GAZE_SIM_THRESH:
        if gazeDist > GAZE_DIST_THRESH:
            if envMag > ENV_CHANGE_THRESH:
                decision = "fixWithHeadMove"
            else:
                decision = "SmoothPursuit"
        else:
            if envMag > ENV_CHANGE_THRESH:
                decision = "HeadPursuit"
            else:
                decision = "fixation"

    if decision == "":
        decision = "None"


    return decision, [[patchDist.item(), patchDistAvg, gazeDist, envMag]]

    
