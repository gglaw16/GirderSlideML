#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 13:17:51 2019

@author: gwenda
"""

import girder as g
import pdb

gc = g.get_gc()
item_id = '5d24bdd370aaa9038e6d1cb4'
name = 'ROI'
image_id = '5d28bf0770aaa9038e6d3c3e'


annotation_id = g.get_annotation_id_from_name(item_id, name, gc)

#pdb.set_trace()
g.add_image_to_annotation(annotation_id, image_id, gc)
