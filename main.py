# This code has been developed by Ashkan Nejad at Royal Dutch Visio and Univeristy Medical Center Groningen

import numpy as np
from sklearn.model_selection import train_test_split

from modules.eventDetector import suffle_trainAndTest as shuffle_trainAndTest_RF
from modules.eventDetector import trainAndTest as trainAndTest_RF
from modules.eventDetector import pred_detector as pred_RF
from modules.preprocess import data_balancer, divider
from modules.GiW import GiWPrep
from config import VALID_METHOD, OUT_DIR


# PREPARING DATASET
ds_x, ds_y = GiWPrep()


###### ANALYSIS


# Preprocess
ds_x = np.asarray(ds_x, dtype="object")
ds_y = np.asarray(ds_y, dtype="object")

if VALID_METHOD == "LOO": # for leave-one-out validation

    ds_x, ds_y = data_balancer(ds_x, ds_y)
    
    f1s_sample = []
    f1s_event = []
    for p in range(1, len(ds_y)):

        x_test = ds_x[p]
        y_test =  ds_y[p]
        x_train = np.array(ds_x)
        x_train = np.delete(ds_x, p, 0)
        y_train = np.array(ds_y)
        y_train = np.delete(ds_y, p, 0)
        f1_ie, f1_is = trainAndTest_RF(x_train, y_train, x_test, y_test, VALID_METHOD)
        f1s_sample.append(f1_ie)
        f1s_event.append(f1_is)

elif VALID_METHOD == "TTS": # for train/test split

    # train/test split procedure
    # ds_x, ds_y = data_balancer(ds_x, ds_y)
    
    X_train, Y_train, X_test, Y_test = divider(ds_x, ds_y)
    
    X_train, Y_train = data_balancer(X_train, Y_train)

    f1_ie, f1_is = trainAndTest_RF(X_train, Y_train, X_test, Y_test, VALID_METHOD)


elif VALID_METHOD == "NV": # for prediction using trained model

    preds = pred_RF(ds_x, ds_y, OUT_DIR+'/models/RF_lblr6.sav')

else:
    print("incorrect validation method in the configuration file")
    
    ds_x, ds_y = data_balancer(ds_x, ds_y)
    #performing complete of labels and samples
    shuffle_trainAndTest_RF(ds_x, ds_y)

    print("")
