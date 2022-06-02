from requests import patch
from scipy.spatial import distance
import numpy as np
from modules.PatchSimNet import perdict

from config import PATCH_SIM_THRESH, GAZE_DIST_THRESH, ENV_CHANGE_THRESH, PATCH_SIZE, LAMBDA, PATCH_PRIOR_STEPS
import modules.patchContent as patchContent


def  eventDetector(patch1, patch2, pastPatchDist, envMag, gazeCoord1, gazeCoord2, patchSimNet_params):
    
    # patchDist = patchContent.compare_old(patch1, patch2) #compute the patch content similarity
    inp = np.zeros((2, PATCH_SIZE, PATCH_SIZE))
    inp[0,:,:] = patch1
    inp[1,:,:] = patch2
    patchDist = perdict(inp, patchSimNet_params) 
    patchDistAvg = ((1-LAMBDA)*patchDist.item() + LAMBDA*(pastPatchDist/PATCH_PRIOR_STEPS))
    gazeDist = distance.euclidean(gazeCoord1, gazeCoord2) #compute the gaze location change

    

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

    
