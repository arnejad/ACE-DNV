INP_DIR = '../sample/' # the input directory containing the video and gaze signal
OUT_DIR = INP_DIR+'res/'
CLOUD_FORMAT = False
VIDEO_SIZE = [1088, 1080]  # [width, height]
GAZE_ERROR = [17, 0] # [x, y]
# PATCH_SIZE = 100
PATCH_SIZE = 64
VISUALIZE = False
PATCH_SIM_THRESH = 50
GAZE_DIST_THRESH = 4
ENV_CHANGE_THRESH = 0.6
LAMBDA = 0.3
PATCH_PRIOR_STEPS = 3

POSTPROCESS_WIN_SIZE = 6