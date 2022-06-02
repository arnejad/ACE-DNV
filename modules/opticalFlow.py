#This function runs the optical flow estimation algorithm. 
# It is based on Gunnar Farneback's algorithm which is explained in "Two-Frame Motion Estimation Based on Polynomial Expansion" by Gunnar Farneback in 2003.
import cv2 as cv
import numpy as np

def opticalFlow(frame1, frame2):
    flow = cv.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

    return mag, ang


def headChange(imu):
    #using the Gyro signal
    gyro = imu[:,1:4]
    
    #normalize each signal independently
    gyro = np.abs(gyro)
    gyro[:,0] = (gyro[:,0] - np.min(gyro[:,0])) / (np.max(gyro[:,0]) - np.min(gyro[:,0]))
    gyro[:,1] = (gyro[:,1] - np.min(gyro[:,1])) / (np.max(gyro[:,1]) - np.min(gyro[:,1]))
    gyro[:,2] = (gyro[:,2] - np.min(gyro[:,2])) / (np.max(gyro[:,2]) - np.min(gyro[:,2]))

    mag = gyro[:,0] + gyro[:,1] + gyro[:,2]
    return mag
    