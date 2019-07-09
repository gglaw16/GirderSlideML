#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:19:26 2019

@author: gwenda
"""

import girder as g


gc = g.get_gc()
image_id = '5d24bdd370aaa9038e6d1cb4'
name = 'ROI'

annotation_id = g.get_annotation_id_from_name(image_id, name, gc)

center, width, height = g.get_annotation_loc_from_id(annotation_id, gc)

print(center)
print(width)
print(height)