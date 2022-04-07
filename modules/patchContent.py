# This part is made using the code publicly provided by Ali Farahani
# https://github.com/a-m-farahani/RandomFerns


import numpy as np
import random
import cv2 as cv
from sklearn.cluster import KMeans
from scipy.spatial import distance




def compare_old(patch1, patch2):

    sift = cv.SIFT_create()

    if (min(patch1.shape)==0) or (min(patch2.shape)==0):
        print("no content detected")
        # return "None"
        return 0


    kps1, descs1 = sift.detectAndCompute(patch1,None)
    kps2, descs2 = sift.detectAndCompute(patch2,None)
    
    if (descs1 is None) or (descs2 is None):
        print("no content detected")
        # return "None"
        return 0


    #concatinate the features
    allDescs = np.empty((0, 128), int)
    allDescs = np.append(allDescs, np.array(descs1), axis=0)
    allDescs = np.append(allDescs, np.array(descs2), axis=0)


    kmeans = KMeans(n_clusters=2, random_state=0).fit(allDescs)
    lbls1 = kmeans.labels_[0:len(descs1)]
    lbls2 = kmeans.labels_[len(descs1):(len(descs1)+len(descs2))]

    hist1 = np.histogram(lbls1, bins=[0,1,2])
    hist2 = np.histogram(lbls2, bins=[0,1,2])

    patchDist = distance.cosine(hist1[0], hist2[0])

    return patchDist
    