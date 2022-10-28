# This code has been developed by Ashkan Nejad at Royal Dutch Visio and Univeristy Medical Center Groningen
# personal conda env: dfvo2

from enum import unique
from os import listdir, path
from os.path import isfile, join
import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.spatial import distance
from scipy import ndimage

from modules.wildMove import VOAnalyzer
from modules.timeMatcher import timeMatcher
import modules.visualizer as visual
from modules.PatchSimNet import pred_all as patchNet_predAll
from NEED import train as NEED_train
from config import INP_DIR, OUT_DIR, VISUALIZE, VIDEO_SIZE, CLOUD_FORMAT, GAZE_ERROR, DATASET, ACTIVITY_NUM, ACIVITY_NAMES, LABELER
from modules.eventDetector import eventDetector as eventDetector
from modules.decisionMaker import execute as run_need
from modules.preprocess import preprocessor

########## DATA PREPARATION


def zScore_norm(featSet):
    #normalize data (z-score) 
    means = np.mean(featSet, axis=0)
    std = np.std(featSet, axis=0)
    featSet = (featSet-means)/std
    mins = np.min(featSet, axis=0)
    featSet = featSet+ np.abs(mins)
    return featSet



if DATASET == "VisioRUG":

    # list the files inside the input directory
    subFiles = [f for f in listdir(INP_DIR) if isfile(join(INP_DIR, f))]

    # find the video in the input directory
    foundVids = [s for s in subFiles if '.mp4' in s] 
    if len(foundVids) > 1:  raise Exception("The input directory contains more than one mp4 file")
    vidPath = INP_DIR+'/'+foundVids[0]

    # checking if gaze.csv exists
    if path.exists(INP_DIR+'/gaze.csv'): 
        gazePath = INP_DIR+'/gaze.csv'
    elif path.exists(INP_DIR+'/gaze_positions.csv'):
        gazePath = INP_DIR+'/gaze_positions.csv'
    else:
        raise Exception("Could not find gaze.csv or gaze_positions.csv file")

    # checking if timestamp.csv exists
    if not path.exists(INP_DIR+'/world_timestamps.csv'): raise Exception("Could not find the world_timestamps.csv file")
    timestampPath = INP_DIR+'/world_timestamps.csv'

    #read timestamps file
    tempRead = np.genfromtxt(timestampPath, delimiter=',')

    if CLOUD_FORMAT:
        timestamps = tempRead[1:, 2]
    else:
        timestamps = tempRead[1:, 0]

    #read gaze files
    tempRead = np.genfromtxt(gazePath, delimiter=',')

    if ~CLOUD_FORMAT:
        gazes = tempRead[1:,[0,3,4]]
        #the corrdinate origin is bottom left
        gazes[:,2] = 1 - gazes[:,2]
        gazes = gazes * [1, VIDEO_SIZE[1], VIDEO_SIZE[0]]
    else:
        gazes = tempRead[1:, 2:]

    # check if imu data exists.
    if not path.exists(INP_DIR+'/imu_data.csv'): raise Exception("Could not find the imu_data.csv file")
    imuPath = INP_DIR+'/imu_data.csv'

    imu = np.genfromtxt(imuPath, delimiter=',') #read the imu signal
    imu = imu[1:, :]
    imu = np.delete(imu, 1,1)

    ###### PREPROCESS
    gazeMatch = np.array(timeMatcher(timestamps, gazes))
    gazeMatch[:,1] = gazeMatch[:,1]+GAZE_ERROR[0]
    gazeMatch[:,2] = gazeMatch[:,2]+GAZE_ERROR[1]


    imuMatch = np.array(timeMatcher(timestamps, imu))

    envMotion = VOAnalyzer(imuMatch)

    visual.gazeCoordsPlotSave(gazeMatch)


elif DATASET == "GiW":
    # list the files inside the input directory
    subFiles = listdir(INP_DIR)
    subFiles.pop()
    # initial structure checks
    if len(subFiles) < 1:
        raise Exception("Data does not follow the Gaze-in-the-Wild structure.")

    if not path.exists(INP_DIR+'/Extracted_Data'):
        raise Exception("Data is not in the Gaze-in-the-Wild structure. Could not find Extracted_Data folder!")

    activityName = ACIVITY_NAMES[ACTIVITY_NUM-1]
    #filtering participants whom did not participate in activity in question
    # participants = [s for s in subFiles if path.exists(INP_DIR + s + '/' + str(ACTIVITY_NUM))] 

    #filtering participants whom did participate in activity in question and have been labeled by the chosen labeler
    participants = [s for s in subFiles if path.exists(INP_DIR + '/Extracted_Data/' + activityName + '/Labels/' + \
        'PrIdx_'+s+'_TrIdx_' + str(ACTIVITY_NUM) + '_Lbr_' + str(LABELER) + '.mat')] 

    # participants = [s for s in subFiles if path.exists(s)]

    trainset_x = []
    trainset_y = []

    # for p in range(len(participants)):  #change into read all samples TODO
    for p in range(4):

        # video path
        vidPath = INP_DIR + participants[p] + '/' + str(ACTIVITY_NUM) + '/world.mp4'

        # find the activity name
        

        # load the data
        processData = scipy.io.loadmat(INP_DIR + '/Extracted_Data/' + activityName + '/ProcessData_cleaned/' + \
        'PrIdx_'+participants[p]+'_TrIdx_' + str(ACTIVITY_NUM) + '.mat')
        
        # load the labels
        labels = scipy.io.loadmat(INP_DIR + '/Extracted_Data/' + activityName + '/Labels/' + \
        'PrIdx_'+participants[p]+'_TrIdx_' + str(ACTIVITY_NUM) + '_Lbr_'+ str(LABELER) +'.mat')

        labels = np.array(labels['LabelData']['Labels'][0])
        
        frames = processData['ProcessData']['ETG'][0,0][0,0][5][0]
        

        # finding the indices that match with frames
        frames, indcs = np.unique(frames, return_index=True)

        # time points
        T = np.squeeze(np.array(processData['ProcessData']['T'][0,0]))
       


        # take the gazes out
        gazes = np.array(processData['ProcessData']['ETG'][0,0][0,0][8] * processData['ProcessData']['ETG'][0,0][0,0][0])
        
        # load the environment
        # envChanges = VOAnalyzer(returnDist=False)

        visod = np.loadtxt(INP_DIR+participants[p] + "/"+ str(ACTIVITY_NUM) + "/visOdom.txt", delimiter=' ')

        envChanges = visod[:,5]

        frameRange = np.loadtxt(INP_DIR+participants[p] + "/"+ str(ACTIVITY_NUM) + "/range.txt", delimiter=' ')
        startFrame = frameRange[0]+1
        endFrame = frameRange[1]

        # if not all envChanges were computed we trim until where available
        rm_indcs = np.where(frames >= len(envChanges)+startFrame)
        indcs = np.delete(indcs, rm_indcs[0])
        frames = np.delete(frames, rm_indcs[0])
        gazeEndTrimmer = np.min(rm_indcs)


        # if we started from a specific frame remove the previous ones
        rm_indcs = np.where(frames < startFrame)
        indcs = np.delete(indcs, rm_indcs[0])
        frames = np.delete(frames, rm_indcs[0])
        gazeStartTrimmer = np.max(rm_indcs[0])

        
        # match the gazes that fall into frames
        gazeMatch = gazes[indcs]

        # reload gazed in [0, 1] 
        gazes = np.array(processData['ProcessData']['ETG'][0,0][0,0][8])

        # keep the labels that correspond to a frame
        labels = labels[0]
        labels = np.squeeze(labels)
        
        lblMatch = labels[indcs]
        TMatch = T[indcs]

        envChanges = envChanges[frames[:-1]-(int(startFrame)-1)]


        # through our timestamps in the gaze in the beginning
        gazes = gazes[T>TMatch[0]]
        labels = labels[T>TMatch[0]]
        T = T[T>TMatch[0]]

        # through our timestamps in the gaze in the end
        gazes = gazes[T<TMatch[-1]]
        labels = labels[T<TMatch[-1]]
        T = T[T<TMatch[-1]]


        if not path.exists(INP_DIR + participants[p] + '/' + str(ACTIVITY_NUM) +'/patchDists.csv'): 
            patchDists = patchNet_predAll(vidPath, gazeMatch, frames)
            np.savetxt(INP_DIR + participants[p] + '/' + str(ACTIVITY_NUM) +'/patchDists.csv', patchDists, delimiter=',')
        else:
            patchDists = np.loadtxt(INP_DIR + participants[p] + '/' + str(ACTIVITY_NUM) +'/patchDists.csv', delimiter=',')

        patchDists = np.transpose(np.array(patchDists))

        feats,lbls = preprocessor(gazes, patchDists, envChanges, T, TMatch, labels, lblMatch)

        # if p == 0:
        #     trainset_x = [feats]
        #     trainset_y = [lbls]
        # else:
        #     trainset_x = np.append(trainset_x, [feats], axis=2)
        #     trainset_y = np.append(trainset_y, [lbls], axis=1)

        trainset_x.append(feats)
        trainset_y.append(lbls)

        print("Data successfully loaded for participant " + str(participants[p]))



###### ANALYSIS


# fig, axs = visual.knowledgePanel_init()    #initiallization of knowledge panel

# compute patch similarites


# Preprocess

# feats,lbls = preprocessor(gazes, patchDists, envChanges, T, TMatch, labels, lblMatch)

run_need(feats, lbls)


# get environment changes

lblSet = labels #sample-based

rmidcs_nans = np.where(lblSet == 0)


# remove blinks
rmidcs_blinks = np.where(lblSet == 4)

rmidcs = np.concatenate((rmidcs_nans, rmidcs_blinks), axis=1)

################### ML method feature creation

# featSet = np.column_stack((patchDists[:-1], gazes, envChanges[:-1,:]))
featSet = np.column_stack((patchDists, gazeMatch[:-1,:], envChanges))

lblSet = np.delete(lblSet, rmidcs)
featSet = np.delete(featSet, rmidcs,0)
frames = np.delete(frames, rmidcs)



np.savetxt(OUT_DIR+'feats.csv', featSet, delimiter=',')
np.savetxt(OUT_DIR+'lbls.csv', featSet, delimiter=',')
np.savetxt(OUT_DIR+'frames.csv', frames, delimiter=',')


# rmidcs_sac = np.where(lblSet == 3)

# lblSet = np.delete(lblSet, rmidcs_sac)
# featSet = np.delete(featSet, rmidcs_sac,0)



# plt.hist(lblSet, bins=np.arange(6))
# plt.show()
res = run_need(featSet, lblSet)


res, gts = NEED_train(featSet, lblSet)


####################### Rule-based

gazeMatch = gazeMatch + np.abs(np.min(gazeMatch, axis=0))
gazeDists = np.linalg.norm(gazeMatch[:-1] - gazeMatch[1:], axis=1) #compute the gaze location change
gazeDists = ndimage.median_filter(gazeDists, size=10)

lblSet = np.delete(lblSet, rmidcs)
gazeDists = np.delete(gazeDists, rmidcs)
patchDists = np.delete(patchDists, rmidcs)
envChanges = np.delete(envChanges, rmidcs)


# visual.knowledgePanel_update(axs, None, np.column_stack((patchDists[:-1], gazeDists, envChanges[:-1])))
# fig.show()


res = eventDetector(patchDists, gazeDists,envChanges, lblSet)

matches = len(np.where(res == np.transpose(lblSet)[0])[0])
print(matches/len(lblSet))

np.savetxt(OUT_DIR+'res.csv', res, delimiter=',')

# us, iss = np.unique(labels, return_index=True)



print(matches)
# Pass the extracted knowledge to make the final desicion

#knowledge panel visualization
# if VISUALIZE:
#     if (min(nxtPatch.shape)>0):
#         visual.knowledgePanel_update(axs, nxtPatch, finalRes[:,1:])
#         plt.pause(0.0000001)

#     # Press Q on keyboard to  exit
#     if cv.waitKey(25) & 0xFF == ord('q'):
#         break
    
# finalEvents = postprocess(finalEvents)
# cap.release()
# cv.destroyAllWindows()

# f = open(OUT_DIR+'res.csv', 'w')
# writer = csv.writer(f)
# writer.writerow(finalEvents)
# f.close()

# np.savetxt(OUT_DIR+'gazeMatch.csv', gazeMatch, delimiter=',')