import cv2 as cv
from matplotlib.pyplot import axis
import numpy as np
from modules.patchExtractor import patchExtractor

from config import PRE_OPFLOW, INP_DIR,VIDEO_SIZE

def cosine_similarity (a,b):
    if np.dot(a, b) == 0:
        return 0
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def opticalFlow_old(frame1, frame2):
    flow = cv.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

    return mag, ang


def imuAnalyzer(imu):
    #using the Gyro signal
    gyro = imu[:,1:4]
    
    #normalize each signal independently
    gyro = np.abs(gyro)
    gyro[:,0] = (gyro[:,0] - np.min(gyro[:,0])) / (np.max(gyro[:,0]) - np.min(gyro[:,0]))
    gyro[:,1] = (gyro[:,1] - np.min(gyro[:,1])) / (np.max(gyro[:,1]) - np.min(gyro[:,1]))
    gyro[:,2] = (gyro[:,2] - np.min(gyro[:,2])) / (np.max(gyro[:,2]) - np.min(gyro[:,2]))

    mag = gyro[:,0] + gyro[:,1] + gyro[:,2]
    return mag
    

def goWithTheFlow(flow_u, flow_v, gazeCoord):
    x = int(gazeCoord[0])
    y = int(gazeCoord[1])
    segmentX = [x]
    segmentY = [y]

    visit_map = np.zeros(VIDEO_SIZE)
    seg = np.zeros(VIDEO_SIZE)
    visit_map[x,y] = 1
    visit_map[x,y] = 1
    sum_u = flow_u[x,y]
    sum_v = flow_v[x,y]
    
    traved = 0

    while traved < len(segmentY):
        x = segmentX[traved]
        y = segmentY[traved]

        # adding candid neighbors
        for x_t in [-1, 0, 1]:
            for y_t in [-1, 0, 1]:
                if ((x+x_t) >= 1088) or ((x+x_t) < 0):
                    continue
                if ((y+y_t) >= 1080) or ((y+y_t) < 0):
                    continue

                if not visit_map[y+y_t, x+x_t]:
                    sim = cosine_similarity([sum_u/(traved+1), sum_v/(traved+1)], [flow_u[y+y_t, x+x_t], flow_v[y+y_t, x+x_t]])
                    visit_map[y+y_t, x+x_t] = 1
                    if(sim >0.912):
                        segmentX.append(x+x_t)
                        segmentY.append(y+y_t)
                        seg[y+y_t, x+x_t]=1
                        sum_u = sum_u + flow_u[y+y_t, x+x_t]
                        sum_v = sum_v + flow_v[y+y_t, x+x_t]
        
        traved = traved + 1

    # sub_u = np.delete(flow_u, [segmentY, segmentX])
    # bg_avg_u = np.mean(flow_u[np.where(seg==0)])
    # bg_avg_v = np.mean(flow_v[np.where(seg==0)])

    return [sum_u/traved, sum_v/(traved+1)]
    #  [bg_avg_u, bg_avg_v]



def OFAnalyzer(frameNum, gazeCoord1, gazeCoord2):
    frameNum = frameNum-1
    if PRE_OPFLOW:
        filename =  '0'*(5-len(str(frameNum))) + str(frameNum)
        flow_u = np.genfromtxt(INP_DIR+"optic/"+filename+"_flow_u.csv", delimiter=',')
        flow_v = np.genfromtxt(INP_DIR+"optic/"+filename+"_flow_v.csv", delimiter=',')
    else:
        print("Computing Optical Flow not connected yet!") #TODO

    flow_u_patch = patchExtractor(flow_u, gazeCoord2)
    flow_v_patch = patchExtractor(flow_v, gazeCoord2)

    meanGazeFlow = [np.mean(flow_u_patch), np.mean(flow_v_patch)]
    sameBehaveCat = np.zeros(VIDEO_SIZE)
    for i in range(VIDEO_SIZE[0]):      #TODO change to algebric operation
        for j in range(VIDEO_SIZE[1]):
            sim = cosine_similarity(meanGazeFlow, [flow_u[i,j], flow_v[i,j]])
            if sim > 0.912:
                sameBehaveCat[i,j] = 1
    
    portion = sum(sameBehaveCat)/(VIDEO_SIZE[0]*VIDEO_SIZE[1])

    return meanGazeFlow, portion
    

    # atten_uv = goWithTheFlow(flow_u, flow_v, gazeCoord2)
    # return atten_uv, 


    
    # bg_uv

    gaze_atten_sim = cosine_similarity(gazeCoord2-gazeCoord1, atten_uv)
    gaze_bg_sim = cosine_similarity(gazeCoord2-gazeCoord1, bg_uv)



def VOAnalyzer(returnDist=False):
    #TODO: connect DF-VO here
    #reading DFVO results for now

    visod = np.loadtxt(INP_DIR+"1/1/visOdom.txt", delimiter=' ')
    # orient_dist = cosine_similarity(visod[i, 4:7], visod[i-1, 4:7])

    if returnDist:
        # orients = np.sum(np.abs(visod[:-1,4:7] - visod[1:, 4:7]), axis=1)
        # 2*atan2(norm({q1,q2,q3}),
        orients = 2*np.arctan(np.linalg.norm(visod[:, 4:7], axis=1))
    else:
        # orients = np.column_stack((visod[:-1,4:7], visod[1:, 4:7]))
        orients = visod[:,4:7]

    # orient_dist = np.linalg.norm(visod[i, 4:7])
    return orients

    


