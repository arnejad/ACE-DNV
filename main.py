# This code has been developed by Ashkan Nejad at Royal Dutch Visio and Univeristy Medical Center Groningen
# personal conda env: dfvo2

from os import listdir, path
from os.path import isfile, join
import csv
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import scipy.io
import h5py

from modules.wildMove import VOAnalyzer
from modules.postprocessor import postprocess
from modules.timeMatcher import timeMatcher
from modules.patchExtractor import patchExtractor
from modules.eventDetector import eventDetector
import modules.visualizer as visual
from modules.PatchSimNet import create_network as createSimNet
from config import INP_DIR, OUT_DIR, VISUALIZE, VIDEO_SIZE, CLOUD_FORMAT, GAZE_ERROR, PATCH_PRIOR_STEPS, DATASET, ACTIVITY_NUM, ACIVITY_NAMES


##### DATA PREPARATION


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

    #filtering participants whom did not participate in activity in question
    participants = [s for s in subFiles if path.exists(INP_DIR + s + '/' + str(ACTIVITY_NUM))] 
    # participants = [s for s in subFiles if path.exists(s)]

    for p in range(len(participants)):  #change into read all samples TODO
        vidPath = INP_DIR + participants[p] + '/' + str(ACTIVITY_NUM) + '/world.mp4'
        activityName = ACIVITY_NAMES[ACTIVITY_NUM-1]
        processData = mat = scipy.io.loadmat(INP_DIR + '/Extracted_Data/' + activityName + '/ProcessData_cleaned/' + \
        'PrIdx_'+participants[p]+'_TrIdx_' + str(ACTIVITY_NUM) + '.mat')

        labels = mat = scipy.io.loadmat(INP_DIR + '/Extracted_Data/' + activityName + '/Labels/' + \
        'PrIdx_'+participants[p]+'_TrIdx_' + str(ACTIVITY_NUM) + '_Lbr_1.mat')

        labels = np.array(labels['LabelData']['Labels'])

        gazes = np.array(processData['ProcessData']['ETG'][0,0][0,0][8] * processData['ProcessData']['ETG'][0,0][0,0][0])

        # matching gazes with frames by finding unique values
        u, indcs = np.unique(np.array(processData['ProcessData']['ETG'][0,0][0,0][5]), return_index=True)
        gazeMatch = gazes[indcs]
        
        startFrame = np.amax(np.array(processData['ProcessData']['ETG'][0,0][0,0][5]))
        endFrame = np.amax(np.array(processData['ProcessData']['ETG'][0,0][0,0][5]))

        # envMotion = VOAnalyzer(imuMatch)

        print("Data successfully loaded")





###### ANALYSIS
f = 1 #frame counter
cap = cv.VideoCapture(cv.samples.findFile(vidPath)) #prepare the target video
ret, frame1 = cap.read() #read a frame
prvFrame = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)   #color conversion to grayscale
# prvPatch = patchExtractor(prvFrame, gazeMatch[0][1:])
prvPatch = patchExtractor(prvFrame, gazeMatch[0])

finalRes = np.array([[0,0,0,0]])
finalEvents = np.array([""])

fig, axs = visual.knowledgePanel_init()    #initiallization of knowledge panel

patchSimNet_params = createSimNet()

while(1):
    ret, frame2 = cap.read()
    if (not ret) or (f==len(gazeMatch)):
        print('No frames grabbed!')
        break
    nxtFrame = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    nxtPatch = patchExtractor(nxtFrame, gazeMatch[f])

    print(f)
    if f < 3900:
        f = f+1
        continue
    # magF, angF = opticalFlow(prvFrame, nxtFrame)  #calculating the optical flow in the whole environment

    # Pass the extracted knowledge to make the final desicion

    event, res = eventDetector(prvPatch, nxtPatch, np.sum(finalRes[-PATCH_PRIOR_STEPS:,0]),VOAnalyzer(f), gazeMatch[f-1], gazeMatch[f], patchSimNet_params, f)

    print(event)
    finalRes = np.append(finalRes, res, axis=0) #store all the responses
    finalEvents = np.append(finalEvents, [event]) #store all the responses
    #update the traversing freames
    f = f+1
    prvPatch = nxtPatch
    prvFrame = nxtFrame

    #knowledge panel visualization
    if VISUALIZE:
        if (min(nxtPatch.shape)>0):
            visual.knowledgePanel_update(axs, nxtPatch, finalRes[:,1:])
            plt.pause(0.0000001)

        # Press Q on keyboard to  exit
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    
finalEvents = finalEvents[1:]
finalEvents = postprocess(finalEvents)
cap.release()
cv.destroyAllWindows()

f = open(OUT_DIR+'res.csv', 'w')
writer = csv.writer(f)
writer.writerow(finalEvents)
f.close()

np.savetxt(OUT_DIR+'gazeMatch.csv', gazeMatch, delimiter=',')