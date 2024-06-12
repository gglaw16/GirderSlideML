# Upload initiali positive and negative mask to girder item:
# priming the pump. One time use.

#import sys
#import pdb
#import scipy.misc
#import random
#import csv
import cv2
#import math
#import json
#import os
#import glob
#import shutil
import numpy as np
#from pprint import pprint
#from utils import *
#import net_utils
import girder as g


    
if __name__ == '__main__':
    gc = g.get_gc()
    item_id = "5915da13dd98b578723a09c8"
    
    masks = g.get_image_file(gc,item_id,'masks.png')
    dont_care_flip = np.invert(masks[...,0])

    masks = np.dstack((masks[...,1],masks[...,1],masks[...,1],dont_care_flip))
    
    cv2.imwrite('masksnew.png',masks)
    
    gc.uploadFileToItem(item_id, 'masksnew.png')

    


