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

fig,ax=plt.subplots(figsize=(20,5))
plt.plot(gazes[:,1]) 
plt.plot(gazes[:,2]) 

if not path.exists(OUT_DIR): mkdir(OUT_DIR)
if not path.exists(OUT_DIR+'/figs/'): mkdir(OUT_DIR+'/figs/')

plt.savefig(OUT_DIR+'/figs/gazeCoords.png')
plt.close()
###### ANALYSIS
f = 1 #frame counter
cap = cv.VideoCapture(cv.samples.findFile(vidPath))
ret, frame1 = cap.read()
prvFrame = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
prvPatch = patchExtractor(prvFrame, gazeMatch[0][1:])

finalRes = []
while(1):
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    nxtFrame = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    nxtPatch = patchExtractor(nxtFrame, gazeMatch[f][1:])

    magF, angF = opticalFlow(prvFrame, nxtFrame)  #calculating the optical flow in the whole environment

    res = eventDetector(prvPatch, nxtPatch, magF, angF, gazeMatch[f-1][1:], gazeMatch[f][1:])
    print(res)
    finalRes.append(res)

    f = f+1
    prvPatch = nxtPatch
    prvFrame = nxtFrame
    if VISUALIZE:
        if (min(nxtPatch.shape)>0):
            cv.imshow('Frame', prvPatch)
        
        # Press Q on keyboard to  exit
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    

cap.release()
cv.destroyAllWindows()

f = open('res.csv', 'w')
writer = csv.writer(f)
writer.writerow(finalRes)
f.close()
