# DATA VARIABLES
DATASET = "GiW-selected"
# INP_DIR = '/home/ashdev/samples/004'      # the input directory containing the video and gaze signal
INP_DIR = '/media/ashdev/Expansion/data/GiW/'
LABELER = 5
ACTIVITY_NUM = 1            # only for gaze-in-the-wild dataset
ACIVITY_NAMES = ["Indoor_Walk", "Ball_Catch", "Tea_Making", "Visual_Search"]
OUT_DIR = INP_DIR+'res/'
CLOUD_FORMAT = False        #Set to to True if the data has been uploaded to cloud and downloaded (for Pupil devices)

ET_FREQ = 300

# REPRESENTAION
VISUALIZE = True           #Set to True if live ploting of knowledge is wanted. Be aware that it increases execution time significantly.


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