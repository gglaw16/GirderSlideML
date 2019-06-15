import random
import os
import numpy as np
import math
import cv2
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.filters import maximum_filter
import pdb
import torch
from torch.autograd import Variable
import torch.nn as nn
import girder as g



def ensuredir(path):
    if path == "" or os.path.isdir(path):
        return
    ensuredir(os.path.split(path)[0])
    os.makedirs(path)





def get_recursive_filenames(dir_path, extension):
    filenames =[]
    for name in os.listdir(dir_path):
        path = os.path.join(dir_path, name)
        if os.path.isdir(path):
            filenames += get_recursive_filenames(path, extension)
        elif os.path.isfile(path) and os.path.splitext(path)[1] == extension:
            filenames.append(path)
    return filenames
                


    

def pylaw2(pdf, radius, sample_count, margin=0):
    """
    An alternative to pylaw.sample
    Find peaks.  Keep neighboring peaks a minimim distance away.
    greedy algorithm
    margin:  avoid sampling the margin.
    returns (points, scores)
    """
    pdf = pdf.copy()
    points = []
    # Also return the response
    scores = []
    while (len(points) < sample_count):
        (y,x) = np.unravel_index(np.argmax(pdf, axis=None), pdf.shape)
        max_val = pdf[y,x]
        if max_val == 0:
            break        
        ymin = max(y-radius, 0)
        ymax = min(y+radius, pdf.shape[0])
        xmin = max(x-radius,0)
        xmax = min(x+radius, pdf.shape[1])
        pdf[ymin:ymax, xmin:xmax] = 0
        if x < margin or x >= pdf.shape[1]-margin or \
           y < margin or y >= pdf.shape[0]-margin:
            continue
        points.append((y,x))
        scores.append(max_val)
    return np.array(points), np.array(scores)
        

# I used this to watch how classifications changed during training.
def print_output(out_tensor):
    num = out_tensor.shape[0]
    for i in range(num):
        sys.stdout.write('%0.3f,'%out_tensor[i,0,0,0])
    print("")
    print("")
    #sys.stdout.flush()

    
# 1d for now: For debugging
def print_variable(v):
    num = v.size()[0]
    for i in range(num):
        sys.stdout.write('%0.3f, '%v[i])
    print("")
    #sys.stdout.flush()

# batch is a tensor
def upload_batch_to_girder(batch, name):
    images = []
    batch_np = batch.numpy()
    shape = batch_np.shape
    # Loop over the images.
    for i in range(shape[0]):
        # split up the components into threes.
        for j in range(0, shape[1], 3):
            image = batch_np[i, j:j+3, ...]
            if image.shape[0] == 3:
                image = image * 255.0
                image = np.clip(image, 0, 255)
                image = image.astype(np.uint8)
                images.append(np.moveaxis(image, 0, 2))
    g.upload_images(images, name, '5abe931f3f24e537140e6ea6')




# network stuff.

def shock_weights(net):
    for layer in net.layers:
        if type(layer).__name__ == 'Conv2d':
            shock_layer_weights(layer)

def shock_layer_weights(layer):
    if type(layer).__name__ != 'Conv2d':
        print("--- Warning: Can only shock Conv layers")
    shape = layer.weight.size()
    noise = torch.randn(shape)
    noise.normal_(std=0.1)
    layer.weight.data.add_(noise)

def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

# compute the distance of a point from a line segment.
def point_distance_from_segment(pt, end0, end1):
    px = pt[0] - end0[0]
    py = pt[1] - end0[1]
    vx = end1[0] - end0[0]
    vy = end1[1] - end0[1]
    mag = math.sqrt(vx*vx + vy*vy)
    vx = vx / mag
    vy = vy / mag
    dx =  vx*px + vy*py
    dy = -vy*px + vx*py

    if dx > 0:
        if dx > mag:
            dx = dx - mag
        else:
            dx = 0
    return math.sqrt(dx*dx + dy*dy)
            


def randrange(min, max):
    return (random.random() * (max - min)) + min


def gaussian(x, std, m=0.0):
    """
    gaussian(0, std) is always 0 here.
    """
    return math.e**(-0.5*(float(x-m)/std)**2)

def normalized_gaussian(x, std, m=0.0):
    """
    normalized so area is 1
    """
    return 1/(math.sqrt(2*math.pi)*std)*math.e**(-0.5*(float(x-m)/std)**2)


def next_conv(net, idx):
    #layers = net.layers + net.post
    while idx < len(net.layers):
        layer = net.layers[idx]
        if type(layer).__name__ == 'Conv2d':
            return layer, idx+1
        idx = idx + 1
    return None, -1



def reset_batch_norm(net):
    """
    Set the batch norm layers back to identity by changing weights and bias of conv layer.
    """
    net.eval()
    #layers = net.layers + net.post
    for layer in net.layers:
        if type(layer).__name__ == 'Conv2d':
            # keep a reference to the preceding convolutional layer.
            conv_layer = layer
        if isinstance(layer, nn.BatchNorm2d):
            conv_layer.bias.requires_grad = False
            conv_layer.weight.requires_grad = False

            # Move running mean to bias
            conv_layer.bias -= layer.running_mean
            layer.running_mean.zero_()

            # just a test
            #layer.running_var *= 4
            #conv_layer.bias *= 2
            #conv_layer.weight *= 2

            #tmp = torch.ones(len(layer.running_var)) * 4.0
            tmp = layer.running_var.reciprocal()

            layer.running_var *= tmp
            tmp = tmp.sqrt()
            conv_layer.bias *= tmp           

            # A complex way to broadcast "backward". 
            shape = conv_layer.weight.shape
            tmp   = tmp.expand(shape[::-1]) # [::-1] Reverses the tuple.
            tmp   = tmp.permute(range(len(shape))[::-1])
            conv_layer.weight *= tmp

            conv_layer.bias.requires_grad = True
            conv_layer.weight.requires_grad = True


    

def transfer_net_weights(in_net, out_net):
    """
    Copy all the weights we can from one network architecture 
    into a second network architecture.
    """
    in_idx = 0
    in_layer, in_idx = next_conv(in_net, in_idx)
    out_idx = 0
    out_layer, out_idx = next_conv(out_net, out_idx)
    while in_idx > 0 and out_idx > 0:
        in_shape = in_layer.weight.shape
        out_shape = out_layer.weight.shape
        print(in_shape, out_shape)
        if in_shape == out_shape:
            out_layer.bias = in_layer.bias
            out_layer.weight = in_layer.weight
            print("%d -> %d copied"%(in_idx, out_idx))
        # Special case for the multi-res low-high merge NIN
        #if out_shape[1] == in_shape[1] * 2:
        #    num = in_shape[1]
        #    pdb.set_trace()
        #    out_layer.weight.data[:,0:num,...] = in_layer.weight.data
        #    out_layer.weight.data[:,num:2*num,...] = in_layer.weight.data
        #    print("   duplicated")
        in_layer, in_idx = next_conv(in_net, in_idx)
        out_layer, out_idx = next_conv(out_net, out_idx)


# set operation on a list.
# changes a.
def union(a, b):
    for e in b:
        if not e in a:
            a.append(e)
    return a



