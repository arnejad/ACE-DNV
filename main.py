# This code has been developed by Ashkan Nejad at Royal Dutch Visio and Univeristy Medical Center Groningen

import numpy as np

from modules.classifier import suffle_trainAndTest as shuffle_trainAndTest_RF
from modules.classifier import trainAndTest as trainAndTest_RF
from modules.classifier import pred_detector as pred_RF
from modules.preprocess import data_balancer_Random, divider_randomInnerEventSplit, data_balancer_nonRandom
from modules.utils import finalReportPrint
from modules.GiW import GiWPrep
from config import VALID_METHOD, OUT_DIR, EXP


# PREPARING DATASET
ds_x, ds_y = GiWPrep()


###### ANALYSIS


# Preprocess
ds_x = np.asarray(ds_x, dtype="object")
ds_y = np.asarray(ds_y, dtype="object")
f1s_sample = []
f1s_event = []
supp_sample = []
supp_event = []
if VALID_METHOD == "LOO": # for leave-one-out validation
    
    
    for p in range(1, len(ds_y)):

        x_test = ds_x[p]
        y_test =  ds_y[p]
        x_train = np.array(ds_x)
        x_train = np.delete(ds_x, p, 0)
        y_train = np.array(ds_y)
        y_train = np.delete(ds_y, p, 0)
        x_train, y_train = data_balancer_Random(x_train, y_train)
        f1_ie, f1_is, supp_e, supp_s = trainAndTest_RF(x_train, y_train, x_test, y_test, VALID_METHOD)
        f1s_sample.append(f1_is)
        f1s_event.append(f1_ie)
        supp_sample.append(supp_s)
        supp_event.append(supp_e)
    

    
  
elif VALID_METHOD == "TTS": # for train/test split


    for rep in range(10):

        # train/test split procedure
        ds_x, ds_y = data_balancer_nonRandom(ds_x, ds_y)
        
        X_train, Y_train, X_test, Y_test = divider_randomInnerEventSplit(ds_x, ds_y)

        # X_train, Y_train = data_balancer_Random(X_train, Y_train)

        f1_ie, f1_is, supp_e, supp_s = trainAndTest_RF(X_train, Y_train, X_test, Y_test, VALID_METHOD)
        f1s_sample.append(f1_is)
        f1s_event.append(f1_ie)
        supp_sample.append(supp_s)
        supp_event.append(supp_e)



elif VALID_METHOD == "NV": # for prediction using trained model

    preds = pred_RF(ds_x, ds_y, OUT_DIR+'/models/RF.sav')

else:
    print("validation method is not set properly in the configuration file")
    print("AD-HOC: Shuffle and learning. CAUTION: event-level scores will be meaningless in this experiment")
    
    ds_x, ds_y = divider_randomInnerEventSplit(ds_x, ds_y)


    shuffle_trainAndTest_RF(ds_x, ds_y)

#calculated weighted averages
f1_samp_avg = np.sum(f1s_sample*(supp_sample/np.sum(supp_sample, axis=0)), axis=0)
f1_event_avg =  np.sum(f1s_event*(supp_event/np.sum(supp_event, axis=0)), axis=0)


# Final report
finalReportPrint(EXP, f1_samp_avg, f1_event_avg)
# np.savetxt(OUT_DIR+'LOO_f1S.csv', f1s_sample, delimiter=",")
# np.savetxt(OUT_DIR+'LOO_f1E.csv', f1s_event, delimiter=",")
