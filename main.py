# This code has been developed by Ashkan Nejad at Royal Dutch Visio and Univeristy Medical Center Groningen

from os import listdir
from os import path
from os.path import isfile, join
import csv
from queue import Empty
import numpy as np
import cv2 as cv

from modules.opticalFlow import opticalFlow
from modules.gazeMatcher import gazeMatcher
from modules.patchExtractor import patchExtractor
from modules.eventDetector import eventDetector
from config import VISUALIZE

inpDir = '../sample' # the input directory containing the video and gaze signal

##### DATA PREPARATION
# list the files inside the input directory
subFiles = [f for f in listdir(inpDir) if isfile(join(inpDir, f))]

# find the video in the input directory
foundVids = [s for s in subFiles if '.mp4' in s] 
if len(foundVids) > 1:  raise Exception("The input directory contains more than one mp4 file")
vidPath = inpDir+'/'+foundVids[0]

# checking if gaze.csv exists
if not path.exists(inpDir+'/gaze.csv'): raise Exception("Could not find the gaze.csv file")
gazePath = inpDir+'/gaze.csv'

# checking if timestamp.csv exists
if not path.exists(inpDir+'/world_timestamps.csv'): raise Exception("Could not find the world_timestamps.csv file")
timestampPath = inpDir+'/world_timestamps.csv'

#read timestamps file
tempRead = np.genfromtxt(timestampPath, delimiter=',')
timestamps = tempRead[1:, 2]

#read gaze files
tempRead = np.genfromtxt(gazePath, delimiter=',')
gazes = tempRead[1:, 2:]



###### PREPROCESS
gazeMatch = np.array(gazeMatcher(timestamps, gazes))
gazeMatch[:,1] = gazeMatch[:,1]+16
gazeMatch[:,2] = gazeMatch[:,2]+60

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
    # minMAG = magF.min()
    # maxMAG = magF.max()
    # minANG = angF.min()
    # maxANG = angF.max()
    # print(f"\rminMAG: {minMAG}, maxMAG: {maxMAG}, minANG: {minANG}, maxANG: {maxANG}")
    # tmp.append([minMAG, maxMAG, minANG, maxANG])
    # magP, angP = opticalFlow(prvPatch, nxtPatch)  #calculating the optical flow in the frame
    res = eventDetector(prvPatch, nxtPatch, magF, angF, gazeMatch[f-1][1:], gazeMatch[f][1:])
    print(res)
    finalRes.append(res)

    f = f+1
    prvPatch = nxtPatch
    if VISUALIZE:
        if (min(nxtPatch.shape)>0):
            cv.imshow('Frame', nxtPatch)
        
        # Press Q on keyboard to  exit
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    

cap.release()
cv.destroyAllWindows()

f = open('res.csv', 'w')
writer = csv.writer(f)
writer.writerow(finalRes)
f.close()
