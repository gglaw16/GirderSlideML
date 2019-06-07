# I do not know the units of the cutout center arg.
# this will test it.
# also,  how does it crop?


import sys
import ipdb
import scipy.misc
import random
import csv
import cv2
import math
import json
import os
import glob
import shutil
import numpy as np
from pprint import pprint
from utils import *
import net_utils
import girder as g


    
if __name__ == '__main__':
    center = (0,15000)
    gc = g.get_gc()
    item_id = "5915d969dd98b578723a09c2"

    #ipdb.set_trace()
    g.get_image(item_id, level=6, cachedir="./tmp")
    
    """
    im = g.get_image_cutout(gc, item_id, center, 500, 500, scale=1)
    cv2.imwrite("level0.png", im)
    im = g.get_image_cutout(gc, item_id, center, 500, 500, scale=0.5)
    cv2.imwrite("level1.png", im)
    im = g.get_image_cutout(gc, item_id, center, 500, 500, scale=0.25)
    cv2.imwrite("level2.png", im)
    im = g.get_image_cutout(gc, item_id, center, 500, 500, scale=0.125)
    cv2.imwrite("level3.png", im)
    im = g.get_image_cutout(gc, item_id, center, 500, 500, scale=0.0625)
    cv2.imwrite("level4.png", im)
    """


    

