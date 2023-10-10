# This code has been developed by Ashkan Nejad at Royal Dutch Visio and Univeristy Medical Center Groningen
# personal conda env: dfvo2

from enum import unique
from os import listdir, path
from os.path import isfile, join, isdir
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
# from NEED import train as NEED_train
from config import INP_DIR, OUT_DIR, VISUALIZE, VIDEO_SIZE, CLOUD_FORMAT, DATASET, ACTIVITY_NUM, ACIVITY_NAMES, LABELER
from modules.eventDetector import eventDetector_new as train_RF
from modules.eventDetector import pred_detector as pred_RF
from modules.eventDetector import trainAndTest as NEED_TrainAndTest
from modules.decisionMaker import execute as run_need
from modules.preprocess import preprocessor, data_stats, data_balancer

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


    # list the folders in the directory
    recs = [f for f in listdir(INP_DIR) if isdir(join(INP_DIR, f))]
    ds_x = []
    ds_y = []

    for r in recs:
        r = '3'
        directory = join(INP_DIR, r)

        # list the files inside the input directory
        subFiles = [f for f in listdir(directory) if isfile(join(directory, f))]

        # find the video in the input directory
        if 'world.mp4' in subFiles: 
            vidPath = join(directory, 'world.mp4')
        else:
            raise Exception("The input directory contains more than one mp4 file")


        # checking if gaze.csv exists
        if 'gaze.csv' in subFiles: 
            gazePath = directory+'/gaze.csv'
        elif 'gaze_positions.csv' in subFiles:
            gazePath = directory+'/gaze_positions.csv'
        else:
            raise Exception("Could not find gaze.csv or gaze_positions.csv file")

        # checking if timestamp.csv exists
        if not  'world_timestamps.csv' in subFiles: raise Exception("Could not find the world_timestamps.csv file")
        timestampPath = directory+'/world_timestamps.csv'

        #read timestamps file
        tempRead = np.genfromtxt(timestampPath, delimiter=',')

        if CLOUD_FORMAT:
            timestamps = tempRead[1:, 2]
        else:
            timestamps = tempRead[1:, 0]

        #read gaze files
        tempRead = np.genfromtxt(gazePath, delimiter=',')

        if ~CLOUD_FORMAT:
            gazes = tempRead[1:,[3,4]]
            #the corrdinate origin is bottom left
            gazes[:,1] = 1 - gazes[:,1]
            gazes = gazes * [VIDEO_SIZE[1], VIDEO_SIZE[0]]
            # tempGazeX = gazes[:,0]
            # tempGazeY = gazes[:,1]
            # gazes = np.column_stack((tempGazeY, tempGazeX))
        else:
            gazes = tempRead[1:, 2:]

        T = tempRead[1:, 0]

        # check if imu data exists.
        # if not path.exists(INP_DIR+'/imu_data.csv'): raise Exception("Could not find the imu_data.csv file")
        # imuPath = INP_DIR+'/imu_data.csv'

        # imu = np.genfromtxt(imuPath, delimiter=',') #read the imu signal
        # imu = imu[1:, :]
        # imu = np.delete(imu, 1,1)

        frames = tempRead[1:, 1]
        frames -= np.min(frames)

        frames, indcs = np.unique(frames, return_index=True)

        gazeMatch = gazes[indcs]
        TMatch = T[indcs]

        # imuMatch = np.array(timeMatcher(timestamps, imu))
        visod = np.loadtxt(join(directory, "visOdom.txt"), delimiter=' ')
        headRot = visod[:,5]
        bodyMotion = visod[:,1:3]

        if not path.exists(join(directory, 'patchDists.csv')): 
            patchDists = patchNet_predAll(vidPath, gazeMatch, frames)
            np.savetxt(join(directory, 'patchDists.csv'), patchDists, delimiter=',')
        else:
            patchDists = np.loadtxt(join(directory, 'patchDists.csv'), delimiter=',')

        patchDists = np.transpose(np.array(patchDists))

        np.savetxt(OUT_DIR + 'gazeMatch.csv', gazeMatch, delimiter=',')

        feats,_ = preprocessor(gazes, patchDists, headRot, bodyMotion, T, TMatch, None, None)
        
        ds_x.append(feats)

        break


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

    ds_x = []
    ds_y = []

    for p in range(len(participants)):  #change into read all samples TODO
    # for p in range(2):

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
        # headRot = VOAnalyzer(returnDist=False)

        visod = np.loadtxt(INP_DIR+participants[p] + "/"+ str(ACTIVITY_NUM) + "/visOdom.txt", delimiter=' ')

        headRot = visod[:,5]

        frameRange = np.loadtxt(INP_DIR+participants[p] + "/"+ str(ACTIVITY_NUM) + "/range.txt", delimiter=' ')
        startFrame = frameRange[0]+1
        endFrame = frameRange[1]

        # if not all headRot were computed we trim until where available
        rm_indcs = np.where(frames >= len(headRot)+startFrame)
        indcs = np.delete(indcs, rm_indcs[0])
        frames = np.delete(frames, rm_indcs[0])
        # gazeEndTrimmer = np.min(rm_indcs)


        # if we started from a specific frame remove the previous ones
        rm_indcs = np.where(frames < startFrame)
        indcs = np.delete(indcs, rm_indcs[0])
        frames = np.delete(frames, rm_indcs[0])
        # gazeStartTrimmer = np.max(rm_indcs[0])

        
        # match the gazes that fall into frames
        gazeMatch = gazes[indcs]

        # reload gazed in [0, 1] 
        gazes = np.array(processData['ProcessData']['ETG'][0,0][0,0][8])

        # keep the labels that correspond to a frame
        labels = labels[0]
        labels = np.squeeze(labels)
        
        lblMatch = labels[indcs]
        TMatch = T[indcs]

        headRot = headRot[frames[:-1]-(int(startFrame)-1)]


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

        feats,lbls = preprocessor(gazes, patchDists, headRot, T, TMatch, labels, lblMatch)

        #temp delete gaze followings
        rmInd = np.where(lbls==3)[0]

        lbls = np.delete(lbls, rmInd[6000:])
        feats = np.delete(feats, rmInd[6000:], axis=0)

        print("number of fixations " + str(len(np.where(lbls==0)[0])) + ", saccades " + str(len(np.where(lbls==2)[0])) + ", gazeP " + str(len(np.where(lbls==1)[0])) + ", gazeF " + str(len(np.where(lbls==3)[0])))
        

        
        ds_x.append(feats)
        ds_y.append(lbls)

        # np.savetxt(OUT_DIR+'feats.csv', feats, delimiter=',')
        # np.savetxt(OUT_DIR+'lbls.csv', lbls, delimiter=',')
        # np.savetxt(OUT_DIR+'frames.csv', frames, delimiter=',')
        print("Data successfully loaded for participant " + str(participants[p]))

elif DATASET == "GiW-selected":
    # list the files inside the input directory
   
    ds_x = []
    ds_y = []

    rec_list = np.loadtxt(INP_DIR +'files_lblr5.txt', delimiter=',', dtype='str')

    for p in range(len(rec_list)):  #change into read all samples TODO
    # for p in range(2):
        
        activityName = rec_list[p, 2] 
        activityNum  = rec_list[p, 1]
        participantNum  = rec_list[p, 0]
        
        # video path
        vidPath = INP_DIR + participantNum + '/' + str(activityNum) + '/world.mp4'

        # load the data
        processData = scipy.io.loadmat(INP_DIR + '/Extracted_Data/' + activityName + '/ProcessData_cleaned/' + \
        'PrIdx_'+participantNum+'_TrIdx_' + str(activityNum) + '.mat')
        
        # load the labels
        labels = scipy.io.loadmat(INP_DIR + '/Extracted_Data/' + activityName + '/Labels/' + \
        'PrIdx_'+participantNum+'_TrIdx_' + str(activityNum) + '_Lbr_'+ str(LABELER) +'.mat')

        labels = np.array(labels['LabelData']['Labels'][0])
        
        frames = processData['ProcessData']['ETG'][0,0][0,0][5][0]
        

        # finding the indices that match with frames
        frames, indcs = np.unique(frames, return_index=True)

        # time points
        T = np.squeeze(np.array(processData['ProcessData']['T'][0,0]))
       


        # take the gazes out
        gazes = np.array(processData['ProcessData']['ETG'][0,0][0,0][8] * processData['ProcessData']['ETG'][0,0][0,0][0])
        
        # load the environment
        # headRot = VOAnalyzer(returnDist=False)

        visod = np.loadtxt(INP_DIR+participantNum + "/"+ str(activityNum) + "/visOdom.txt", delimiter=' ')

        headRot = visod[:,5]
        bodyMotion = visod[:,1:3]


        frameRange = np.loadtxt(INP_DIR+participantNum + "/"+ str(activityNum) + "/range.txt", delimiter=' ')
        startFrame = frameRange[0]+1
        endFrame = frameRange[1]

        # if not all headRot were computed we trim until where available
        rm_indcs = np.where(frames >= len(headRot)+startFrame)
        indcs = np.delete(indcs, rm_indcs[0])
        frames = np.delete(frames, rm_indcs[0])
        # gazeEndTrimmer = np.min(rm_indcs)


        # if we started from a specific frame remove the previous ones
        rm_indcs = np.where(frames < startFrame)
        indcs = np.delete(indcs, rm_indcs[0])
        frames = np.delete(frames, rm_indcs[0])
        # gazeStartTrimmer = np.max(rm_indcs[0])

        
        # match the gazes that fall into frames
        gazeMatch = gazes[indcs]

        # reload gazed in [0, 1] 
        gazes = np.array(processData['ProcessData']['ETG'][0,0][0,0][8])

        # keep the labels that correspond to a frame
        labels = labels[0]
        labels = np.squeeze(labels)
        
        lblMatch = labels[indcs]
        TMatch = T[indcs]

        headRot = headRot[frames[:-1]-(int(startFrame)-1)]
        bodyMotion = bodyMotion[frames[:-1]-(int(startFrame)-1)]

        # through our timestamps in the gaze in the beginning
        gazes = gazes[T>TMatch[0]]
        labels = labels[T>TMatch[0]]
        T = T[T>TMatch[0]]

        # through our timestamps in the gaze in the end
        gazes = gazes[T<TMatch[-1]]
        labels = labels[T<TMatch[-1]]
        T = T[T<TMatch[-1]]


        if not path.exists(INP_DIR + participantNum + '/' + str(activityNum) +'/patchDists.csv'): 
            patchDists = patchNet_predAll(vidPath, gazeMatch, frames)
            np.savetxt(INP_DIR + participantNum + '/' + str(activityNum) +'/patchDists.csv', patchDists, delimiter=',')
        else:
            patchDists = np.loadtxt(INP_DIR + participantNum + '/' + str(activityNum) +'/patchDists.csv', delimiter=',')

        patchDists = np.transpose(np.array(patchDists))

        feats,lbls = preprocessor(gazes, patchDists, headRot, bodyMotion, T, TMatch, labels, lblMatch)

        #temp delete gaze followings
        # rmInd = np.where(lbls==3)[0]
        # lbls = np.delete(lbls, rmInd)
        # feats = np.delete(feats, rmInd, axis=0)

        #temp delete fixation
        # rmInd = np.where(lbls==0)[0]
        # lbls = np.delete(lbls, rmInd)
        # feats = np.delete(feats, rmInd, axis=0)

        # #temp delete gaze p
        # rmInd = np.where(lbls==1)[0]
        # lbls = np.delete(lbls, rmInd)
        # feats = np.delete(feats, rmInd, axis=0)

        # lbls[lbls == 2] = 0
        # lbls[lbls == 3] = 1

        print("number of fixations " + str(len(np.where(lbls==0)[0])) + ", saccades " + str(len(np.where(lbls==2)[0])) + ", gazeP " + str(len(np.where(lbls==1)[0])) + ", gazeF " + str(len(np.where(lbls==3)[0])))
        
        # lbls[np.where(lbls==3)] = 1
        
        ds_x.append(feats)
        ds_y.append(lbls)

        np.savetxt(OUT_DIR+'feats_p' + str(participantNum) + '_a' + str(activityNum)+ '_l' + str(LABELER) +  '.csv', feats, delimiter=',')
        np.savetxt(OUT_DIR+'lbls_p' + str(participantNum) + '_a' + str(activityNum) + '_l' + str(LABELER) +'.csv', lbls, delimiter=',')
        np.savetxt(OUT_DIR+'frames_p' + str(participantNum) + '_a' + str(activityNum) +  '.csv', frames, delimiter=',')
        np.savetxt(OUT_DIR+'gazes_p' + str(participantNum) + '_a' + str(activityNum) +  '.csv', gazeMatch, delimiter=',')
        print("Data successfully loaded for participant " + str(participantNum) + " task: " + str(activityNum))
    

###### ANALYSIS


# fig, axs = visual.knowledgePanel_init()    #initiallization of knowledge panel

# compute patch similarites


# Preprocess

ds_x, ds_y = data_balancer(ds_x, ds_y)

ds_x = np.array(ds_x, dtype=object); 

if ds_y: ds_y = np.array(ds_y, dtype=object)


train_RF(ds_x, ds_y)


# for prediction using saved model
# preds = pred_RF(ds_x, ds_y, OUT_DIR+'/models/RF_lblr6.sav')


#saving predictions
# np.savetxt(OUT_DIR + 'events.csv', preds, delimiter=',')

