import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt


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
        # gaze_feats = extract_gaze_features(x,y, window, latency, length, labels)
        gaze_feats, lbls = extract_gaze_features_single(x,y, labels)
        return gaze_feats, lbls

def extract_gaze_features(x,y, windows, latency, length, targets):
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

def extract_gaze_features_single(x,y, targets):
    '''
    Creates a training and a target tensor based on the number of
    windows, the latency offset, number of targets, and gaze coordinates
    '''
    latency = 1000/ET_FREQ #in ms
    offset = 0
    ini = 0
    tr_tensor  = np.zeros((len(x)-ini, 2)) #num X sets of features
    tgt_tensor = np.zeros(len(targets)-ini,)
    for i in range(ini, len(x)-1):
        diff_x = x[i] - x[i+1]
        diff_y = y[i] - y[i+1]
        ampl   = np.math.sqrt(diff_x**2 + diff_y**2)
        time   = ((1)*latency)/1000
        #saving speed
        tr_tensor[i-ini][0] = ampl/time
        #saving direction % window
        tr_tensor[i][1] = np.math.atan2(diff_y, diff_x)
    tgt_tensor = targets
    return tr_tensor, tgt_tensor

def extract_head_features(x):
    '''
    Creates a training and a target tensor based on the number of
    windows, the latency offset, number of targets, and gaze coordinates
    '''
    latency = 1000/ET_FREQ #in ms
    length = 1
    stride = 9
    strides = [2**val for val in range(stride)]
    fac = (ET_FREQ * length)/strides[-1]
    windows = [int(np.ceil(i*fac)) for i in strides]
    offset = 0
    ini = int(np.ceil(ET_FREQ * length))
    tr_tensor  = np.zeros((len(x)-ini, 2*len(windows))) #num X sets of features
    for i in range(ini, len(x)):
        for j in range(len(windows)):
            start_pos, end_pos = get_start_end(i,windows[j])
            if start_pos == end_pos:
                continue
            diff_x = x[end_pos] - x[start_pos]
            ampl   = np.math.sqrt(diff_x**2)
            time   = ((end_pos - start_pos)*latency)/1000
            #saving speed
            tr_tensor[i-ini][j] = ampl/time
            #saving direction % window
            tr_tensor[i-ini][j+len(windows)] = np.math.atan(diff_x)
    return tr_tensor

def extract_head_features_single(x):
    '''
    Creates a training and a target tensor based on the number of
    windows, the latency offset, number of targets, and gaze coordinates
    '''

    latency = 1000/ET_FREQ #in ms
    ini = 0
    tr_tensor  = np.zeros((len(x)-ini, 2)) #num X sets of features

    for i in range(ini, len(x)-1):
        diff_x = x[i] - x[i+1]
        ampl   = np.math.sqrt(diff_x**2)
        time   = ((1)*latency)/1000
        #saving speed
        tr_tensor[i-ini][0] = ampl/time
        #saving direction % window
        tr_tensor[i][1] = np.math.atan(diff_x)
    return tr_tensor


def extract_body_features(coords):
    
    dist = np.linalg.norm(coords[:-1] - coords[1:], axis=1)
    vel = dist/(1/ET_FREQ)
    return vel

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

    # check if the signal has more than one dimention
    if len(signal.shape) > 1:
        ynew = []
        for d in range (signal.shape[1]):
            f = interp1d(current_t, signal[:,d], kind='cubic')
            ynew_col = f(target_t)
            if ynew==[]:
                ynew = ynew_col
            else:
                ynew= np.column_stack((ynew, ynew_col))
    else:
        # fit the interpolation
        f = interp1d(current_t, signal, kind='cubic')
        ynew = f(target_t)
        # extract the target timepoints 
        

    return ynew

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def preprocessor(gaze, patchSim, headRot, bodyLoc, gaze_t, frame_t, labels, lblMatch):

    
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
    # patchSim = np.delete(patchSim, rmidcs)
    # headRot = np.delete(headRot, rmidcs)


    if len(lblMatch)-1 in rmidcs :
        rmidcs = np.delete(rmidcs[0], np.where(rmidcs==(len(lblMatch)-1))[1])
        patchSim = np.delete(patchSim, rmidcs)
        headRot =  np.delete(headRot, rmidcs)
        bodyLoc =  np.delete(bodyLoc, rmidcs, axis=0)
        patchSim = patchSim[:-1]
        headRot = headRot[:-1]
        bodyLoc = bodyLoc[:-1,:]
    else:
        patchSim = np.delete(patchSim, rmidcs)
        headRot =  np.delete(headRot, rmidcs)
        bodyLoc =  np.delete(bodyLoc, rmidcs, axis=0)

    betweenFrame_t = (frame_t[:-1] + frame_t[1:]) / 2
    adjustIncs_lower = np.where(gaze_t < betweenFrame_t.min())
    adjustIncs_higher = np.where(gaze_t > betweenFrame_t.max())
    rmidcs = np.concatenate((adjustIncs_lower, adjustIncs_higher), axis=1)
    labels = np.delete(labels, rmidcs)
    gaze = np.delete(gaze, rmidcs, axis=0)
    gaze_t = np.delete(gaze_t, rmidcs)



    # map the labels to our method label
    
    # Fixation = 0 (originally 1),
    labels[labels == 1] = 0; lblMatch[lblMatch == 1] = 0; 
    # Gaze Pursuit = 1 (originally 2),
    labels[labels == 2] = 1; lblMatch[lblMatch == 2] = 1; 
    # Saccade = 2 (originally 3),
    labels[labels == 3] = 2; lblMatch[lblMatch == 3] = 2; 
    # Gaze Following = 3 (originally 5)
    labels[labels == 5] = 3; lblMatch[lblMatch == 5] = 3; 



    
    # print("number of fixations " + str(len(np.where(labels==0)[0])) + ", saccades " + str(len(np.where(labels==2)[0])) + ", gazeP " + str(len(np.where(labels==1)[0])) + ", gazeF " + str(len(np.where(labels==3)[0])))


    
    


    # extract gaze features
    gaze_feat, labels_new = gazeFeatureExtractor(gaze, labels)
    

    # print("number of fixations " + str(len(np.where(labels==0)[0])) + ", saccades " + str(len(np.where(labels==2)[0])) + ", gazeP " + str(len(np.where(labels==1)[0])) + ", gazeF " + str(len(np.where(labels==3)[0])))
    # 
    # interpolate patch similarities and head rotations to the same sampling rate as gaze

    

    # none-gaze-related (body translation and head rotaion) feature extrtaction
    head_feats = extract_head_features_single(headRot)
    body_feat = extract_body_features(bodyLoc)
    body_feat = np.concatenate((body_feat, [body_feat[-1]])) #TODO make more accurate

    # filtering body movement spikes
    # body_feat = median_filter(body_feat, size=20)
    body_feat[np.where(body_feat > np.quantile(body_feat, 0.75))] = np.quantile(body_feat, 0.75)
    body_feat = NormalizeData(body_feat)

    # signal interpolations
    head_feats = interpolate(head_feats, betweenFrame_t, gaze_t)
    body_feat = interpolate(body_feat, betweenFrame_t, gaze_t)
    new_patchSim = interpolate(patchSim, betweenFrame_t, gaze_t)

    
    

    # final_feats = np.concatenate((gaze_feat, head_feats, np.transpose([new_patchSim[300:]])), axis=1)
    final_feats = np.column_stack((gaze_feat, head_feats, body_feat, np.transpose([new_patchSim])))

    return final_feats, labels_new


def data_stats(y):

    lbls = np.squeeze(y)
    lbls = np.concatenate(y)

    plt.hist(lbls, bins = 4)
    plt.show()


# def balancer(x,y):



