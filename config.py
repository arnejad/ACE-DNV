# DATA VARIABLES
DATASET = "GiW-selected"
# DATASET = "VisioRUG"
# INP_DIR = '/home/ashdev/samples/004'      # the input directory containing the video and gaze signal
# INP_DIR = '/media/ashdev/Expansion/data/Walking_indoors/'

INP_DIR = '/media/ash/Expansion/data/GiW/'
CH2STREAM_MODEL_DIR = '/media/ash/Expansion/2ch2stream_notredame.t7'  # Directory to the saved weights for 2ch2stream network

EXP = "1-2"     #two digits: first one experiment number according to paper. second, subconditions
                #Experiment 1: 1)IndoorWalk 2)BallCatch 3)visualSearch
                #Experiment 2: 1)all data, separate GFi and GFo, 2) All data, combined GFi and GFo (similar to what GiW did)
                # Experiment 3: Indoorwalk + ball catch with agreement labels

VALID_METHOD = 'TTS' # choose between 'LOO' for leave one out or "TTS" for train/test split
GFiGFo_Comb = False   #considering gaze fixation and gaze following as same class (class 0). this is the defult. For exp 2-2, it changes to True below.
# LABELER = 5               #labeler number or 'MV' for majority voting or 'AG' for agreements or "NV" for no validation and just testing saved models
ACTIVITY_NUM = 1            # only for gaze-in-the-wild dataset
ACIVITY_NAMES = ["Indoor_Walk", "Ball_Catch", "Tea_Making", "Visual_Search"]
OUT_DIR = INP_DIR+'res/'
CLOUD_FORMAT = False        #Set to to True if the data has been uploaded to cloud and downloaded (for Pupil devices)

ET_FREQ = 300

# REPRESENTAION
VISUALIZE = True           #Set to True if you would like to have Feature Histogram, Confusion matrices, and distribution hist of each epoch


# INPUT DATA INFORMATION
VIDEO_SIZE = [1080, 1088]   # [width, height]
GAZE_ERROR = [17, 0]        # [x, y]
IMU_AVAIL = True            #set to True if IMU signal is available
PRE_OPFLOW = True           #set to True if optic is provided and is not needed to be computed


# FRAMEWORK VARIABLES
PATCH_SIZE = 64
PATCH_SIM_THRESH = 50       #Threshold for similarity in patch content
GAZE_DIST_THRESH = 4        #Threshold for gaze location distance
ENV_CHANGE_THRESH = 0.6     #threshold for environemnt change
LAMBDA = 0.0                #Momentum parameter to regulize patch similarity fluctuation
PATCH_PRIOR_STEPS = 3       # Number of previouse frames taking part in regulazing next patch content similarty
POSTPROCESS_WIN_SIZE = 6    # Size of window running on extracted events for voting.
GWF_SIM_THRESH = 0.91       #Go with the flow flow similarity threshold



if EXP == "1-1":                        #single task: Indoor walk
    RECS_LIST = "files_task1_lblr5.txt"
elif (EXP == "2-1" or EXP == "2-2"):    # all tasks together
    RECS_LIST = "files_all_lblr6.txt"  
elif EXP=="1-2":                        # single task: Ball catch
    RECS_LIST = "files_task2_lblr6.txt"   
elif EXP == "1-3":                       # single task: Visual search
    RECS_LIST = "files_tasks3_lblr6.txt"   
elif EXP == "3":                        # all tasks with agreement labels
    RECS_LIST = "files_ag.txt"   
if EXP == "2-2": GFiGFo_Comb = True
