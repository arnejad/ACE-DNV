# This function extracts the patch arround the gaze location.
from config import PATCH_SIZE

def patchExtractor (frame, gazeCord):
    x = gazeCord[0]
    y = gazeCord[1]
    width = frame.shape[1]
    height = frame.shape[0]

    if (x-(PATCH_SIZE/2) < 0): 
        minX = 0 
    else: 
        minX = round(x-(PATCH_SIZE/2))

    if (x+(PATCH_SIZE/2) > width): 
        maxX = width
    else: 
        maxX = round(x+(PATCH_SIZE/2))

    if (y-(PATCH_SIZE/2) < 0): 
        minY = 0 
    else: 
        minY = round(y-(PATCH_SIZE/2))

    if (y+(PATCH_SIZE/2) > height): 
        maxY = height
    else: 
        maxY = round(y+(PATCH_SIZE/2))

    return frame[minY:maxY, minX:maxX]
    