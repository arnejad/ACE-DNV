import numpy as np
from scipy.interpolate import interp1d

from config import ET_FREQ

# Functions in this file are extracted from work of Elmadjian et al. on https://github.com/elmadjian/OEMC

def gazeFeatureExtractor(gaze, labels):
        '''
        data: dataframe to extract features from
        stride: number of multiscale windows of powers of 2, i.e.:
                window sizes for stride 3 -> 1, 2, 4
                window sizes for stride 5 -> 1, 2, 4, 8, 16 
        target_offset: position of the target in w.r.t. the 
                       last sample of the window. E.g.,
                       an offset of 1 = next sample is the target
                       an offset of -5 = look-ahead of 5 
        '''
        length = 1
        stride = 9
        strides = [2**val for val in range(stride)]
        fac = (ET_FREQ * length)/strides[-1]
        window = [int(np.ceil(i*fac)) for i in strides]
        latency = 1000/ET_FREQ #in ms
        x = gaze[:,0]
        y = gaze[:,1]
        gaze_feats = extract_features(x,y, window, latency, length, labels)
        return gaze_feats

def extract_features(x,y, windows, latency, length, targets):
    '''
    Creates a training and a target tensor based on the number of
    windows, the latency offset, number of targets, and gaze coordinates
    '''
    offset = 0
    ini = int(np.ceil(ET_FREQ * length))
    tr_tensor  = np.zeros((len(x)-ini, 2*len(windows))) #num X sets of features
    tgt_tensor = np.zeros(len(targets)-ini,)
    for i in range(ini, len(x)):
        for j in range(len(windows)):
            start_pos, end_pos = get_start_end(i,windows[j])
            if start_pos == end_pos:
                continue
            diff_x = x[end_pos] - x[start_pos]
            diff_y = y[end_pos] - y[start_pos]
            ampl   = np.math.sqrt(diff_x**2 + diff_y**2)
            time   = ((end_pos - start_pos)*latency)/1000
            #saving speed
            tr_tensor[i-ini][j] = ampl/time
            #saving direction % window
            tr_tensor[i-ini][j+len(windows)] = np.math.atan2(diff_y, diff_x)
        tgt_tensor[i-ini] = targets[i+offset]
    return tr_tensor, tgt_tensor

def get_start_end( i, step):
    '''
    i -> the most recent position
    '''
    end_pos = i
    start_pos = i - step
    if start_pos < 0:
        start_pos = 0
    return start_pos, end_pos


def interpolate(signal, current_t, target_t):

    # fit the interpolation
    f = interp1d(current_t, signal, kind='cubic')

    # extract the target timepoints 
    ynew1 = f(target_t)

    return


def preprocessor(gaze, patchSim, headRot, gaze_t, frame_t, labels, lblMatch):


    # remove blinks and unlabled data #TODO remove also neighboring events

    rmidcs_nans = np.where(labels == 0)
    rmidcs_blinks = np.where(labels == 4) # remove blinks

    rmidcs = np.concatenate((rmidcs_nans, rmidcs_blinks), axis=1)
    labels = np.delete(labels, rmidcs)
    gaze = np.delete(gaze, rmidcs, axis=0)
    gaze_t = np.delete(gaze_t, rmidcs)

    rmidcs_nans = np.where(lblMatch == 0)
    rmidcs_blinks = np.where(lblMatch == 4) # remove blinks
    rmidcs = np.concatenate((rmidcs_nans, rmidcs_blinks), axis=1)
    
    frame_t = np.delete(frame_t, rmidcs)
    
    if len(lblMatch)-1 in rmidcs :
        patchSim = np.delete(patchSim, rmidcs[:-1])
        headRot =  np.delete(headRot, rmidcs[:-1])
    else:
        patchSim = np.delete(patchSim, rmidcs)
        headRot =  np.delete(headRot, rmidcs)
    


    # map the labels to our method label
    
    # Fixation = 0 (originally 1),
    labels[labels == 1] = 0; lblMatch[lblMatch == 1] = 0; 
    # Gaze Pursuit = 1 (originally 2),
    labels[labels == 2] = 1; lblMatch[lblMatch == 2] = 1; 
    # Saccade = 2 (originally 3),
    labels[labels == 3] = 2; lblMatch[lblMatch == 3] = 2; 
    # Gaze Following = 3 (originally 5)
    labels[labels == 5] = 3; lblMatch[lblMatch == 5] = 3; 
    

    # extract gaze features
    gaze_feat, labels_new = gazeFeatureExtractor(gaze, labels)
    



    return gaze_feat, labels_new
    
    # interpolate patch similarities and head rotations to the same sampling rate as gaze

    # betweenFrame_t = (frame_t[:-1] + frame_t[1:]) / 2

    # new_patchSim = interpolate(patchSim, betweenFrame_t, gaze_t)
    # new_headRot = interpolate(headRot, betweenFrame_t, gaze_t)


