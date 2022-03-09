# This function corresponds the gaze signals with the frames based on the caputered timestamps

from cmath import inf

def gazeMatcher(timestamps, gazes):

    res = []
    f=1 #frame inspection pointer (tranverses through timestamp)
    g=1 #gaze inspection pointer (tranverses through gazes)

    while f < len(timestamps):
        inspecF = f
        minVal = inf
        while ((f==inspecF) and (g<len(gazes))):
            dist = abs(timestamps[f] - gazes[g,0])
            if dist <= minVal: #distance is decreasing, then continue travers
                minVal = dist
                g = g+1
            else:
                res.append([timestamps[f], gazes[g-1,1], gazes[g-1,2]])
                f = f+1
        if g==len(gazes): #if no gaze left to assign to frame, repear the final gaze location #TODO: Improve assigning
            res.append([timestamps[f], gazes[g-1,1], gazes[g-1,2]])
            f = f+1
            
    return res