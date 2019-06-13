# Stuf associated with executing the network.



import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import scipy.misc
import cv2
import math
import os
from my_utils import *
import numpy as np
from pprint import pprint





def find_plane(planes, (x,y), precision=0.5):
    for plane in planes:
        if 'bbox' in plane:
            bbox = plane['bbox']
            if bbox[0] > bbox[2]:
                tmp = bbox[0]
                bbox[0] = bbox[2]
                bbox[2] = tmp
            if bbox[1] > bbox[3]:
                tmp = bbox[1]
                bbox[1] = bbox[3]
                bbox[3] = tmp
            # bbox is too big. There can be close by targets.
            # lets make it smaller.
            center = ((bbox[2]+bbox[0])*0.5, (bbox[3]+bbox[1])*0.5)
            radius = ((bbox[2]-bbox[0])*precision/2.0, (bbox[3]-bbox[1])*precision/2.0)
            if abs(x-center[0]) < radius[0] and abs(y-center[1]) < radius[1]:
                return plane
        if 'detection' in plane:
            detection = plane['detection']
            radius = detection['rf_size'] * detection['spacing'] * 0.5
            center = detection['center']
            if abs(x-center[0]) < radius and abs(y-center[1]) < radius:\
               return plane
    return None




def get_debug_heatmap_overlay(img, heatmap):
    fx = img.shape[0] / heatmap.shape[0]
    fy = img.shape[1] / heatmap.shape[1]

    if np.max(heatmap) < 2.0:
        heatmap = heatmap * 255.0

    mask2 = cv2.resize(heatmap,None,fx=fx, fy=fy, interpolation = cv2.INTER_LINEAR)
    dy, dx = mask2.shape
    mask3 = mask2.reshape((dy, dx, 1))
    red_mask = np.concatenate((np.zeros((dy, dx, 2), dtype=np.uint8), mask3), axis=2)
    white_mask = np.concatenate((mask3, mask3, mask3), axis=2)/(255.0 + 128.0)
    
    # combine
    img2 = img * (1.0-white_mask) + red_mask*white_mask
    img3 = img2.astype(np.uint8)
    return img3


def crop_chip(image, center, size):
    hsize = int(size/2)
    xmin = center[0]-hsize
    xmax = xmin + size
    ymin = center[1]-hsize
    ymax = ymin + size
    # crop to image
    xmin = max(xmin,0)
    xmax = min(xmax, image.shape[1])
    ymin = max(ymin,0)
    ymax = min(ymax, image.shape[0])
    if xmin > xmax or ymin > ymax:
        return None
    return (xmin, ymin), image[ymin:ymax, xmin:xmax, ...]


def get_island_bounds(mask):
    """ Return the index bounds of the island.
    """
    # Project to the x axis
    tmp = mask.max(axis=0)
    # argmax returns the first.
    xmin = np.argmax(tmp)
    # reverse it
    tmp = tmp[::-1]  
    xmax = len(tmp) - np.argmax(tmp) - 1
    # Now do the same for the y axis
    tmp = mask.max(axis=1)
    ymin = np.argmax(tmp)
    tmp = tmp[::-1]  
    ymax = len(tmp) - np.argmax(tmp) - 1
    return [xmin,ymin, xmax,ymax]


def get_island_radius(mask, cx, cy):
    """ Return the index bounds of the island.
    """
    # Project to the x axis
    tmp = mask.max(axis=0)
    # argmax returns the first.
    xmin = np.argmax(tmp)
    # reverse it
    tmp = tmp[::-1]  
    xmax = len(tmp) - np.argmax(tmp) - 1
    # Now do the same for the y axis
    tmp = mask.max(axis=1)
    ymin = np.argmax(tmp)
    tmp = tmp[::-1]  
    ymax = len(tmp) - np.argmax(tmp) - 1

    radius = max(xmax-cx, ymax-cy, cx-xmin, cy-ymin)
    return radius


def heatmap_to_detections(heatmap, params, spacing=1, min_score=0.2):
    annotations = {'planes':[]}
    while True:
        (y,x) = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
        score = heatmap[y,x]
        if score < min_score:
            return annotations
        # Isolate the island for this detection
        thresh = max(score/2, min_score)
        _, mask = cv2.threshold(heatmap, thresh, 127, cv2.THRESH_BINARY)
        cv2.floodFill(mask, None, (x,y), 255)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        # Find the bbox for this island
        #bds = get_island_bounds(mask)
        # double the size of the detection (because the net responds only to the center).
        #cx = 0.5*(bds[0]+bds[2])
        #cy = 0.5*(bds[1]+bds[3])
        #bds[0] = 2*bds[0] - cx
        #bds[1] = 2*bds[1] - cy
        #bds[2] = 2*bds[2] - cx
        #bds[3] = 2*bds[3] - cy
        radius = get_island_radius(mask, x, y) * 2
        # double the size of the detection (because the net responds only to the center).
        radius = radius * 3
        
        # Convert heatmap coordinat system to image coordinat system
        hm_spacing = spacing * params['rf_stride']
        cx = x * hm_spacing
        cy = y * hm_spacing
        #bds[0] = bds[0] * hm_spacing
        #bds[1] = bds[1] * hm_spacing
        #bds[2] = bds[2] * hm_spacing
        #bds[3] = bds[3] * hm_spacing
        radius = radius * hm_spacing
        
        radius = max(radius, 30)
        radius = min(radius, 60)
        
        # Add the detection
        annotations['planes'].append({'detection': {'score':score,
                                                    'spacing':spacing,
                                                    'center':(cx,cy),
                                                    'radius' :radius,
                                                    'rf_size':params['rf_size']}})
        
        # mask out this island in the heatmap
        heatmap[mask>127] = 0


def test_large_image(img, net, params, annotations, min_score = 0.2, debug_id="", spacings=[2]):
    """
    debug_id: just keep debugging images separate on disk
    """
    net.eval()

    for spacing in spacings:
        # Shrink the image to the specied resolution.
        small = cv2.resize(img, (0,0), fx=1.0/spacing, fy=1.0/spacing)
    
        net.eval()
        net_out = execute_large_image(net, small, params)
        heatmap = net_out[..., 1]
        ret_heatmap = heatmap.copy()

        if params['debug']:
            heat_overlay = get_debug_heatmap_overlay(small, ret_heatmap);
            cv2.imwrite('debug/heatmap_overlay%s.png'%debug_id, heat_overlay)
            debug_map = heatmap * 255.0
            debug_map = debug_map.astype(np.uint8)
            cv2.imwrite(os.path.join('debug', "map%s_%d.png"%(debug_id,spacing)), debug_map)    

        annotations = heatmap_to_detections(heatmap, params, spacing=spacing,
                                            min_score=min_score)
            
        """
        # assume target fills half the receptive field.
        # This may lead to double detections.
        # TODO: Do contours / islands.
        radius = params['rf_size'] / (params['rf_stride']*4)
        points, scores = pylaw2(heatmap, 15, 500)
        
        # save out chips, and record annotation meta data at the same time.
        # I wonder if iterating over numpy arrays works like this.
        for pt, score in zip(points, scores):
            if score < min_score:
                break
            # So they will be ordered score high to low in the folder.
            # TODO: Fix pylaw2 to return points in standard (x,y) order instead of np order.
            x = int((pt[1]+0.5) * params['rf_stride'] * spacing)
            y = int((pt[0]+0.5) * params['rf_stride'] * spacing)
            # See if we have an existing annotation for this location.
            plane = find_plane(annotations['planes'], (x,y))
            if plane is None:
                annotations['planes'].append({'detection': {'score':score,
                                                            'spacing':spacing,
                                                            'center':(x,y),
                                                            'rf_size':params['rf_size']}})
            else:
                # Skip the detection if it has already been marked as true.
                if not 'bbox' in plane:
                    detection = plane['detection']
                    # keep the detection that has the highest score
                    if score > detection['score']:
                        detection['score'] = score
                        detection['spacing'] = spacing
                        detection['center'] = (x,y)
                        detection['rf_size'] = params['rf_size']
        """
        
    return annotations, ret_heatmap





#=================================================================================
def execute_image(net, image, params):
    """
    Input is np image,  output is np array (components last axis)
    This just manipulates/rformats the input and output.
    """

    # first: reformat image into float from 0 to 1.
    image = image.astype(np.float) / 255.0
    # reformat into batch
    image = np.moveaxis(image, 2, 0)
    shape = image.shape
    image = image.reshape((1,shape[0],shape[1],shape[2]))

    # Now do the pytorch stuff
    image_tensor = torch.from_numpy(image).float()
    image_tensor = image_tensor.cuda(params['gpu'])
    #image_variable = Variable(image_tensor)
    output = net(image_tensor)

    # TODO: Pass through softmax to make the range 0-1
    smax = nn.Softmax(dim=1)
    output = smax(output)
    
    # convert output to an image.
    tmp = output.data.cpu()
    output = tmp.numpy()
    # Return all channels, Put the target channels at the end
    output = output[0,...]
    output = np.moveaxis(output, 0, 2)
    
    return output
                                                                  


#=================================================================================
def execute_large_image(net, image, params):
    """
    Execute the network on a large image.
    The image is normal 3 channel, dtype=npunit8
    It just returns a large numpy array ((YDim, xDim) float32 (range 0-1).
    If the network has a stride, the output will be smaller than the dimensions.
    There is no shrinkage due to convolution because we pad the input.
    """

    # Parameter to grid up the image to execute in pieces.
    panel_size = 116
    # overlap due to convolution
    stride = net.get_rf_stride()
    in_overlap = net.get_rf_size() - stride 
    
    # allocate the output array. Simply divide by 4 because we pad the input.
    in_shape = image.shape
    # Deleay allocating the output until we know how many targets there are
    # out = np.zeros((in_shape[0]/stride, in_shape[1]/stride), dtype=np.float32)
    output_dim_y = int(in_shape[0]/stride)
    output_dim_x = int(in_shape[1]/stride)
    output = None
    
    # Pad the input image because the output shrinks due to convolution.
    in_margin = int(in_overlap/2)
    # allocate
    in_pad = np.ones((in_shape[0]+in_overlap, in_shape[1]+in_overlap, 4), dtype=np.uint8) * 128
    # copy the inpout into the new array
    in_pad[in_margin:in_margin+in_shape[0], in_margin:in_margin+in_shape[1], :] = image
    in_image = in_pad
    # the in_shape is now bigger because of the padding.
    in_shape = in_image.shape
    
    # divide up the output into a grid of panels.
    # These are ideal panel dimensions.
    # THe ones cropped out may be smaller because uneven divisions.
    grid_dim_x = max(int(math.ceil(float(output_dim_x) / panel_size)), 1)
    grid_dim_y = max(int(math.ceil(float(output_dim_y) / panel_size)), 1)
    out_panel_dim_x = int(math.ceil(float(output_dim_x) / grid_dim_x))
    out_panel_dim_y = int(math.ceil(float(output_dim_y) / grid_dim_y))

    print('')
    for grid_y in range(grid_dim_y):
        print("\033[1A %d, %d"%(grid_y, grid_dim_x))
        out_y = grid_y * out_panel_dim_y
        out_dim_y = min(output_dim_y-out_y, out_panel_dim_y)
        in_y = out_y*stride
        in_dim_y = out_dim_y*stride + in_overlap
        for grid_x in range(grid_dim_x):
            out_x = grid_x * out_panel_dim_x
            out_dim_x = min(output_dim_x-out_x, out_panel_dim_x)
            in_x = out_x*stride
            in_dim_x = out_dim_x*stride + in_overlap

            # crop the panel from the padded input image
            in_panel = in_image[in_y:in_y+in_dim_y, in_x:in_x+in_dim_x, :]
            if in_panel.shape[1] < 10 or in_panel.shape[0] < 10:
                continue
            out_panel = execute_image(net, in_panel, params)
            out_panel = out_panel[0:out_dim_y, 0:out_dim_x, :]
 
            if output is None:
                # Allocate the large array
                num_channels = out_panel.shape[2]
                output = np.zeros((output_dim_y, output_dim_x, num_channels), dtype=np.float32)
            
            # copy the output_panel into larger output
            shape = out_panel.shape
            output[out_y:out_y+shape[0], out_x:out_x+shape[1], :] = out_panel
            
    return output

