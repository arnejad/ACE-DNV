from requests import patch
from scipy.spatial import distance
import numpy as np
from modules.PatchSimNet import perdict

from config import PATCH_DIST_THRESH, GAZE_DIST_THRESH, ENV_CHANGE_THRESH, PATCH_SIZE
import modules.patchContent as patchContent


def  eventDetector(patch1, patch2, magF, angF, gazeCoord1, gazeCoord2, patchSimNet_params):
    
    # patchDist = patchContent.compare_old(patch1, patch2) #compute the patch content similarity
    inp = np.zeros((2, PATCH_SIZE, PATCH_SIZE))
    inp[0,:,:] = patch1
    inp[1,:,:] = patch2
    patchDist = perdict(inp, patchSimNet_params) 

    gazeDist = distance.euclidean(gazeCoord1, gazeCoord2) #compute the gaze location change

    magMean = np.mean(magF) #compute shift in environment

    # print(magMean)

    ## final decision
    # if patchDist > PATCH_DIST_THRESH:
    #     if gazeDist > GAZE_DIST_THRESH:
    #         return "PotSAC"
    #     elif gazeDist < GAZE_DIST_THRESH:
    #         if magMean > ENV_CHANGE_THRESH:
    #             return "fixation"
    
    # else:# patchDist < PATCH_DIST_THRESH:
    #     if gazeDist > GAZE_DIST_THRESH:
    #         if magMean > ENV_CHANGE_THRESH:
    #             return "fixWithHead"
    #         else:
    #             return "SmoothPursuit"
    #     else:
    #         if magMean < ENV_CHANGE_THRESH:
    #             return "HeadPursuit"
    #         else:
    #             return "fixation"

    # return "None"
    return [[patchDist, gazeDist, magMean]]

    
