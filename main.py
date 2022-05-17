# This code has been developed by Ashkan Nejad at Royal Dutch Visio and Univeristy Medical Center Groningen

from os import listdir, path, mkdir
from os.path import isfile, join
import csv
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

from modules.opticalFlow import opticalFlow
from modules.gazeMatcher import gazeMatcher
from modules.patchExtractor import patchExtractor
from modules.eventDetector import eventDetector
import modules.visualizer as visual
from modules.PatchSimNet import create_network as createSimNet
from config import INP_DIR, OUT_DIR, VISUALIZE, VIDEO_SIZE, CLOUD_FORMAT, GAZE_ERROR


##### DATA PREPARATION
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
    gazes = gazes * [1, VIDEO_SIZE[0], VIDEO_SIZE[1]]
else:
    gazes = tempRead[1:, 2:]

###### PREPROCESS
gazeMatch = np.array(gazeMatcher(timestamps, gazes))
gazeMatch[:,1] = gazeMatch[:,1]+GAZE_ERROR[0]
gazeMatch[:,2] = gazeMatch[:,2]+GAZE_ERROR[1]

visual.gazeCoordsPlotSave(gazeMatch)

###### ANALYSIS
f = 1 #frame counter
cap = cv.VideoCapture(cv.samples.findFile(vidPath)) #prepare the target video
ret, frame1 = cap.read() #read a frame
prvFrame = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)   #color conversion to grayscale
prvPatch = patchExtractor(prvFrame, gazeMatch[0][1:])

finalRes = np.array([[0,0,0]])

fig, axs = visual.knowledgePanel_init()    #initiallization of knowledge panel

patchSimNet_params = createSimNet()

while(1):
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    nxtFrame = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    nxtPatch = patchExtractor(nxtFrame, gazeMatch[f][1:])

    magF, angF = opticalFlow(prvFrame, nxtFrame)  #calculating the optical flow in the whole environment

    # Pass the extracted knowledge to make the final desicion
    res = eventDetector(prvPatch, nxtPatch, magF, angF, gazeMatch[f-1][1:], gazeMatch[f][1:], patchSimNet_params)
    print(res)
    finalRes = np.append(finalRes, res, axis=0) #store all the responses

    #update the traversing freames
    f = f+1
    prvPatch = nxtPatch
    prvFrame = nxtFrame

    #knowledge panel visualization
    if VISUALIZE:
        if (min(nxtPatch.shape)>0):
            visual.knowledgePanel_update(axs, nxtPatch, finalRes)
            plt.pause(0.0000001)

        # Press Q on keyboard to  exit
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    
plt.show()

cap.release()
cv.destroyAllWindows()

f = open('res.csv', 'w')
writer = csv.writer(f)
writer.writerow(finalRes)
f.close()
