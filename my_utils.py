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



    
#=================================================================================
# From kwcnn
# Not used.  It did not work and I do not feel like debuggin why.

"""
def find_peaks(image, preprocessing=2, iterations=3, bbox_margin=(15, 15)):
    # Convert color image to greyscale
    if len(image.shape) == 3 and image.shape[2] > 1:
        image_grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_grey = image
        
    # Establish kernel used for dilations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Preprocessing step to initially dilate the image
    mask = cv2.dilate(image_grey, kernel, iterations=preprocessing)

    # In an iterative process:
    #     1.) detect the peaks
    #     2.) turn peaks to mask
    #     3.) perfom morphology operations on the mask (open + dilate)
    #     4.) repeat
    for i in range(iterations):
        detected_peaks = detect_peaks(mask)
        mask = np.zeros(mask.shape, dtype=mask.dtype)
        mask[detected_peaks] = 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, (3, 3), iterations=1)
        
    bbox_list = []
    conf_list = []
    mask_ = np.copy(mask).astype(np.uint8)
    
    im2, contour_list, _ = cv2.findContours(mask_, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contour_list:
        conf = np.max([image_grey[y, x] for [[x, y]] in contour])
        conf_list.append(conf / 255.0)
        
        x, y, w, h = cv2.boundingRect(contour)
        if bbox_margin is not None:
            w2 = w // 2
            h2 = h // 2
            cx = x + w2
            cy = y + h2
            minx = cx - bbox_margin[0]
            miny = cy - bbox_margin[1]
            maxx = cx + bbox_margin[0]
            maxy = cy + bbox_margin[1]
        else:
            minx = x
            miny = y
            maxx = x + w
            maxy = y + h
            
        bbox = (minx, miny, maxx, maxy)
        bbox_list.append(bbox)
        
        return mask, bbox_list, conf_list


def detect_peaks(image):
    ""
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)

    Reference:
    http://stackoverflow.com/a/3689710
    ""
    
    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood) == image
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    
    # we create the mask of the background
    background = (image == 0)
    
    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood,
                                       border_value=1)
    
    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask
    detected_peaks = np.logical_xor(local_max, eroded_background)
    
    return detected_peaks


#=================================================================================
# My kwcnn ROC code


# Load data from girder.  Will ahve to remove that bit

# Plot an roc curve from a mission.
# variable size from the ground truth annotations.
# Return psotive and negative annotations to make an ROC view.
# We may skip creating the raster graph in the future.
# imset is a hack to create separte roc curves
def plot_roc3(folder_id, mission_id, truth_annotation_names, imset=None):
    max_x = 5
    max_y = 1
    # for really bad vehicle detector
    #max_x = 20
    #max_y = 0.5
    gc = get_gc()
    
    # for computing total coverage area.
    pixel_total = 0
    gsd = 0.486     # meters
    
    # legacy, get rid of these in the future.
    # create a list of false positives and true positives.
    # simple confidences are all we need.
    true_confidences = []
    false_confidences = []
    
    # Use these instead
    # I want to display the false positives and negatives.
    # To do this, I need the image id, center, shape and confidences.
    mission_false_negative_annotations = []
    mission_false_positive_annotations = []
    
    # load the json file for the mission.
    mission = load_girder_mission(mission_id)
    # loop over the images in the mission
    idx = 0
    for image_info in mission:
        # Hack to get roc curves for subsets of images.
        if imset and not (idx in imset):
            idx += 1
            continue
        idx += 1
        image_id = image_info['image_id']
        print(image_id)
        # add the image area to to coverage total.
        resp = gc.get("item/%s/tiles" % image_id)
        pixel_total = pixel_total + resp['sizeX'] * resp['sizeY']
        
        # Collect all the ground truth annotations
        # into a single list of elements because
        # ground truth can be split between different annotations
        # in an image (heli, chinook) (car,truck)
        true_elements = []
        # loop over the ground truth annotations
        for truth_annotation_name in truth_annotation_names:
            print("%s : %s"%(image_id,truth_annotation_name))
            annot_id = create_or_find_annotation(image_id, truth_annotation_name)
            annot = load_girder_annotation(annot_id)
            for element in annot['elements']:
                e = {'confidence':-1, image_id: image_id}
                e['radius'] = element['width']/2
                e['x'] = element['center'][0]
                e['y'] = element['center'][1]
                true_elements.append(e)

        # match ground truth with detections
        # loop through detections for this image and classify as true or
        # false positives. False positives are simply detection that were
        # on a grouond truth tile.  True positives are ground truth tiles
        # that have at lease one detection.
        # - false p
        # positives are recorded in "false_confidences" array
        # - true positives are stored in "true_elements" array as "found"
        for detection in image_info['image_objects']:
            # for legacy plots
            if not 'in_level' in detection:
                detection['in_level'] = 1
                detection['in_shape'] = [64,64]
                
            # compute the center point of the detection.
            x = 0.5*(detection['corner_points'][0] + detection['corner_points'][1])
            y = 0.5*(detection['corner_points'][2] + detection['corner_points'][3])
            
            # Find if this detection is in the ground truth annotations.
            found = False
            for e in true_elements:
                if abs(e['x']-x) < e['radius'] and abs(e['y']-y) < e['radius']:
                    found = True
                    # to plot the ROC, remember the greatest confidence of
                    # each true positive.
                    if e['confidence'] < detection['confidence']:
                        e['confidence'] = detection['confidence']
                        
            if not found:
                # save in the negative set.
                false_confidences.append(detection['confidence'])
                e = {'confidence':detection['confidence'],
                     'image_id': image_id,
                     'radius': detection['corner_points'][1]-x,
                     'x': x,
                     'y': y,
                     'in_level': detection['in_level'],
                     'in_shape': detection['in_shape']}
                mission_false_positive_annotations.append(e)

        # Now record the ground truth tiles
        # (with detection confidence) for the positive set.
        # A alternative option is just run the ground truth
        # through the detector by themselves.
        for e in true_elements:
            # May get rid of this and use the second
            true_confidences.append(e['confidence'])
            mission_false_negative_annotations.append(e)
            
        # Now generate the raster plot.
        # TODO: matplot lib is archaic. Either use d3 or generate html.
        # convert pixels to square kilometers
        detection_area = pixel_total * gsd * gsd / 1000000
            
        # Compute and plot the curve
        AUC = 0
        save_true = 0
        save_false = 0
        numSamples = 1000
        X = []
        Y = []
        for i in range(numSamples+1):
            threshold = 1.0-(float(i)/numSamples)
            # count the false positives above the threshold
            total = len(false_confidences)
            num = 0
            for conf in false_confidences:
                if conf > threshold:
                    num = num + 1
            false_positive_per_km2 = num / detection_area
                
            # count the true positives above the threshold
            total = len(true_confidences)
            num = 0
            for conf in true_confidences:
                if conf > threshold:
                    num = num + 1
            true_positive_rate = float(num) / total
                
            # crop the graph at x = max_x
            if false_positive_per_km2 > max_x:
                # interpolate to set x = max_x
                k = (max_x-save_false) / (false_positive_per_km2 - save_false)
                false_positive_per_km2 = max_x
                true_positive_rate = save_true + k*(true_positive_rate-save_true)
            if save_false < max_x:
                X.append(false_positive_per_km2)
                Y.append(true_positive_rate)
                # for computing the area under the curve
                if false_positive_per_km2 > save_false:
                    dx = false_positive_per_km2 - save_false
                    AUC = AUC + (dx*(true_positive_rate + save_true)/2)
            save_false = false_positive_per_km2
            save_true = true_positive_rate
        
    # if the curve ended early, extend it for the AUC
    if save_false < max_x:
        dx = max_x - save_false
        AUC = AUC + (dx*save_true)
        
    print("ROC AUC = %0.2f" % AUC)
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #fig.subplots_adjust(top=0.85)
    #ax.set_title('axes title')

    plt.figure().clear()
    
    plt.title('ROC')
    plt.text(0.5, 0.01, ('ROC AUC = %0.2f'% AUC),
             verticalalignment='bottom', horizontalalignment='left',
             #transform=ax.transAxes,
             color='blue', fontsize=15)
        
    plt.legend(loc='lower right')
    plt.xlabel('false positives per square km')
    plt.ylabel('true positive rate')
    
    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    
    plt.plot(X, Y, color='red') #, label="ROC")
    plt.ylim([0,max_y])
    plt.xlim([0,max_x])
    #plt.show()
    file_name = "tmp/roc"
    if imset:
        file_name += ''.join('-'+str(e) for e in imset)
    file_name += ".png"
    print(file_name)
    plt.savefig(file_name)
    girder_image = gc.uploadFileToItem(mission_id, file_name)
    
    # upload the ground truth (true_elements) with confidences.
    # sort the elements by confidence.
    mission_false_negative_annotations = sorted(mission_false_negative_annotations,
                                                key=lambda x: x['confidence'])
    mission_false_positive_annotations = sorted(mission_false_positive_annotations,
                                                key=lambda x: -x['confidence'])

    # save to json file
    roc_data = {'false_negative': mission_false_negative_annotations,
                'false_positive': mission_false_positive_annotations,
                'area': detection_area}
    output_path = './tmp'
    roc_data_path = path.join(output_path, "roc_data.json")
    with open(roc_data_path, 'w') as outfile:
        json.dump(roc_data, outfile)
        
        # upload to girder
        girder_item = gc.createItem(folder_id, 'light box', 'data to generate an roc graph')
        gc.addMetadataToItem(girder_item['_id'], {'VigilantChangeDetection':
                                                  "ROCLightBox"})
        gc.uploadFileToItem(girder_item['_id'], roc_data_path)
        return girder_item['_id']
                
"""



    
