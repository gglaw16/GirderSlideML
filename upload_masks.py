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
    item_id = "5915da6add98b578723a09cb"
    
    negative = cv2.imread('negative-f.png',0)
    positive = cv2.imread('positive-f.png',0)
    unknown = np.ones(negative.shape) *255
    unknown -= (positive + negative)
    bgr = np.dstack((unknown,positive,negative)) # stacks 3 h x w arrays -> h x w x 3
    
    cv2.imwrite('masks.png',bgr)
    
    gc.uploadFileToItem(item_id, 'masks.png')


