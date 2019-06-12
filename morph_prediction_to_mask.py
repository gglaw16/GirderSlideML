#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 12:32:06 2019

@author: gwenda
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('prediction4.png',cv2.IMREAD_GRAYSCALE)

kernel = np.ones((10,10),np.uint8)

plt.figure(figsize=(9,12))
plt.imshow(img,cmap='gray')


img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)[1]


plt.figure(figsize=(9,12))
plt.imshow(img,cmap='gray')



img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


plt.figure(figsize=(9,12))
plt.imshow(img,cmap='gray')

cv2.imwrite('map_f.png', img)