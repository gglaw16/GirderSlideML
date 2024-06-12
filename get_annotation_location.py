#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:19:26 2019

@author: gwenda
"""

import girder as g
import math
import matplotlib.pyplot as plt
import cv2

level = 2

gc = g.get_gc()
item_id = '5d24bdd370aaa9038e6d1cb4'
name = 'gwenda.law'

annotation_id = g.get_annotation_id_from_name(item_id, name, gc)

center, width, height = g.get_annotation_loc_from_id(annotation_id, gc)

spacing = math.pow(2, level)

x = center[0]
y = center[1]
w = width/spacing
h = height/spacing

image = g.get_image_cutout(gc, item_id, (x,y), w, h, scale=1.0/spacing, cache='cache')

im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


plt.imshow(im_rgb)