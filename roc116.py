#!/home/local/KHQ/charles.law/envs/pytorch/bin/python
# sbatch -c 3 -p vigilant --gres=gpu:1 test116.py image.png 2

# Process a list of image filenames
# subsample to a target resolution
# run images through a network
# find peaks for detections.
# load ground truth
# Mark detections as false or true

# generate an roc curve






import sys
import shutil
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# The target path is hard coded.
import fcnn116 as target
import pdb
import scipy.misc
import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv
import random
import time
import math
import os
import numpy as np
from pprint import pprint
from utils import *
import girder as g
# todo: get rid of the misc function references.
import data_json
import net_utils
import json



    
            

#=================================================================================
def load_net(params):
    net = target.net()
    params['rf_size'] = net.get_rf_size()
    params['rf_stride'] = net.get_rf_stride()
    
    filename = os.path.join(params['folder_path'], params['target_group'], 'model.pth')
    if os.path.isfile(filename):
        print("Loading network %s"%filename)
        net.load_state_dict(torch.load(filename, \
                                       map_location=lambda storage, loc: storage))

    if 'transfer_net_filename' in params and os.path.isfile(params['transfer_net_filename']):
        print("TRANSFERING WEIGHTS FROM %s"%params['transfer_net_filename'])
        transfer_net = target_last.net()
        transfer_net.load_state_dict(torch.load(params['transfer_net_filename'], \
                                                map_location=lambda storage, loc: storage))
        transfer_net_weights(transfer_net, net)
        print("Saving " + filename)
        torch.save(net.state_dict(), filename)

    return net



# Return four items necessary to compute an ROC curve:
# A list of true positive detections, A list of false positive detections,
# a list of groundtruth targets and the area of the image is square kilomters.
def process_image_for_roc(img_filename, net, params):
    print(img_filename)
    # This is so each image can save a heatmap.
    # It also gives a way to find the image in girder (fixAnnot.py)
    debug_id = os.path.splitext(img_filename)[0]
    debug_id = debug_id.split('-')[-1]
    
    # Get the groundtruth.
    ensuredir(params['annotation_dir'])
    annot_filename = '%s.json'%(os.path.splitext(img_filename)[0])
    annot_filepath = os.path.join(params['annotation_dir'], annot_filename)
    ensuredir(os.path.split(annot_filepath)[0])
    # If there is no annotation file, assume all detections are false.
    groundtruth = []
    if os.path.isfile(annot_filepath):
        with open(annot_filepath) as f:
            truth = json.load(f)
            for plane in truth['planes']:
                if 'bbox' in plane:
                    plane['score'] = 0.0
                    groundtruth.append(plane)
    # Load the sat image.
    img = cv2.imread(os.path.join(params['image_dir'], img_filename))
    pdb.set_trace()
    # Compute the area of the image to get false alarm per square km
    # assume 0.5 meter GSD
    area = img.shape[0] * img.shape[0] / 4000000.0 
    
    annotations = {'planes':[], 'image_dir':params['image_dir'],
                   'image_filename':img_filename}

    annotations, heatmap = net_utils.test_large_image(img, net, params, annotations,
                                                      min_score=0.0, debug_id=debug_id)
    # The problem with returning the heatmap:  WHat is the spacing?
    # TODO: fix this.
    spacing = 2
    
    # Use the heatmap to get scores for the ground truth. Comparing ground truth to annotation
    # peaks will work but is more complex and can break if it is not done well.
    # get the maximum pixel inside the center of the groundtruth rectangle.
    for plane in groundtruth:
        bbox = plane['bbox']
        # bboxes are too big, cut down in half (quarter area)
        cx = (bbox[2]+bbox[0])*0.5/(params['rf_stride'] * spacing)
        cy = (bbox[3]+bbox[1])*0.5/(params['rf_stride'] * spacing)
        rx = abs((bbox[2]-bbox[0])*0.35)/(params['rf_stride'] * spacing)
        ry = abs((bbox[3]-bbox[1])*0.35)/(params['rf_stride'] * spacing)
        region = heatmap[int(cy-ry):int(cy+ry), int(cx-rx):int(cx+rx)]
        try:
            plane['score'] = np.max(region)
        except:
            pdb.set_trace()
            
    false_positives = []
    true_positives = []
    for detect_plane in annotations['planes']:
        detection = detect_plane['detection']
        truth_plane = net_utils.find_plane(groundtruth, detection['center'], precision=0.6)
        if truth_plane is None:
            false_positives.append(detection)

    print((len(true_positives), len(false_positives), len(groundtruth), area))


    # save just the agregious cases.
    for detect in false_positives:
        score = detect['score']
        if score > 0.85:
            r = detect['rf_size'] * detect['spacing']
            c = detect['center']
            chip = img[int(c[1]-r):int(c[1]+r),int(c[0]-r):int(c[0]+r), :]
            fn = 'debug/false_pos_s%02d_x%d_y%d_%s.png'%(int(score*99),c[0],c[1], debug_id)
            print(fn)
            cv2.imwrite(fn, chip)

    false_negatives = [p for p in groundtruth if p['score'] == 0.0]
    for plane in groundtruth:
        score = plane['score']
        if score < 0.15:
            bbox = plane['bbox']
            c = (int((bbox[0]+bbox[2])*0.5), int((bbox[1]+bbox[3])*0.5))
            chip = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
            fn = 'debug/false_neg_s%02d_x%d_y%d_%s.png'%(int(score*99),c[0],c[1], debug_id)
            print(fn)
            cv2.imwrite(fn, chip)
    
    """
    # save out all results for debugging.
    for detect in false_positives:
        r = detect['rf_size'] * detect['spacing']
        c = detect['center']
        chip = img[int(c[1]-r):int(c[1]+r),int(c[0]-r):int(c[0]+r), :]
        fn = 'debug/false_pos_%d_%d.png'%c
        cv2.imwrite(fn, chip)

    for detect in true_positives:
        r = detect['rf_size'] * detect['spacing']
        c = detect['center']
        chip = img[int(c[1]-r):int(c[1]+r),int(c[0]-r):int(c[0]+r), :]
        fn = 'debug/true_pos_%d_%d.png'%c
        cv2.imwrite(fn, chip)

    for plane in groundtruth:
        bbox = plane['bbox']
        c = (int((bbox[0]+bbox[2])*0.5), int((bbox[1]+bbox[3])*0.5))
        chip = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
        fn = 'debug/truth_%d_%d.png'%c
        cv2.imwrite(fn, chip)

    false_negatives = [p for p in groundtruth if p['score'] == 0.0]
    for plane in false_negatives:
        bbox = plane['bbox']
        c = (int((bbox[0]+bbox[2])*0.5), int((bbox[1]+bbox[3])*0.5))
        chip = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
        fn = 'debug/false_neg_%d_%d.png'%c
        cv2.imwrite(fn, chip)
    pdb.set_trace()
    """
    
    return true_positives, false_positives, groundtruth, area
        


def plot_roc(true_positives, false_positives, groundtruth, detection_area):
    max_x = 5
    max_y = 1
    false_confidences = [d['score'] for d in false_positives]
    true_confidences = [p['score'] for p in groundtruth]
    
    # Do the ROC logic.
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
        num = len([x for x in false_confidences if x > threshold])
        false_positive_per_km2 = num / detection_area
        
        # count the true positives above the threshold
        total = len(groundtruth)
        num = len([x for x in true_confidences if x > threshold])
        true_positive_rate = float(num) / total

        # Cut the curve at the maximum value for x.
        # We need to interpolate to make it correct.
        if false_positive_per_km2 > max_x:
            k = (max_x - save_false) / (false_positive_per_km2 - save_false)
            false_positive_per_km2 = max_x
            true_positive_rate = save_true + (true_positive_rate-save_true)*k
        
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

    try:
        plt.figure().clear()
    except:
        print("Could not clear figure.  DISPLAY?")
              
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
    try:
        plt.show()
    except:
        print("Could not show figure.  DISPLAY?")
    
    file_name = "roc.png"
    plt.savefig(file_name)


def get_image_filenames_recursive(dirname):
    filenames = []
    for name in os.listdir(dirname):
        path = os.path.join(dirname, name)
        if os.path.isdir(path):
            tmp = get_image_filenames_recursive(path)
            # add the subdirectory to the path (but not the top level directory).
            tmp = [os.path.join(name,x) for x in tmp]
            filenames += tmp
        elif os.path.splitext(path)[1] == '.png':
            # leave off the image path
            filenames.append(name)
    return filenames




    
        
if __name__ == '__main__':
    params = {}
    # We get these from the net now
    #params['rf_stride'] = 4
    #params['rf_size'] = 116
    params['min_augmentation_scale'] = 0.6
    params['folder_path'] = '.'
    params['truth_radius'] = 16
    params['ignore_radius'] = 40
    params['gpu'] = 0 #3 # 0
    params['num_epochs'] = 100
    # Batches / Epoch: Load a new image every # batchs
    params['num_batches'] = 50
    # resample batch training images every # cycles
    params['num_minibatches'] = 8
    params['rate'] = 0.05
    params['heatmap_decay'] = 0.2
    params['debug'] = True #False 
    params['target_group'] = 'fcnn116'
    params['max_num_training_images'] = 6000
    # impacts gpu memory usage
    params['input_size'] = 132
    params['batch_size'] = 32

    #params['annotation_dir'] = './test/Shanghai'
    #params['image_dir'] = './test/Shanghai'

    #params['annotation_dir'] = './test/Seshcha'
    #params['image_dir'] = './test/Seshcha'
    
    params['annotation_dir'] = './test'
    params['image_dir'] = './images'
    
    #img_filenames = ["Airports/3857_11_1021_681_20161129_508a508d-196f-4df5-83fa-50fe054ef05c.png"]
    #img_filename = "incomplete/Soltsy/3857_11_1196_615_20161210_1d558ebf-14d0-4681-a004-a0f882f81af7.png"
    #img_filename = "train/Chicago/3857_8_65_95_20170210_15da7492-f98d-470a-8e28-b86aa9484e59.png"
    #img_filename = "train/Belaya/3857_8_201_83_20170120_0e958b30-460a-467e-80d6-2b129dcdb25c.png"

    #img_filenames = get_image_filenames_recursive(params['image_dir'])
    #img_filenames = get_image_filenames_recursive(params['image_dir'])


    # One obvious plane has no response.  What's up?
    #img_filenames = ["Seshcha/3857_12_2427_1320_20170202_fb984fe3-a01b-4b96-9609-24d4b25ee4bd.png"]

    # 100 detections was too low a limit
    #img_filenames = ["Shanghai/3857_10_858_418_20170202_417973b5-2fe8-4dd0-a607-0c1ea9dd5ce6.png"]


    img_filenames = ["Misc/Wonsan_port_20180726.png"]

    
    
    net = load_net(params)
    net.cuda(params['gpu'])

    true_positives = []
    false_positives = []
    groundtruth = []
    total_area_km2 = 0.0
    for img_filename in img_filenames:
        tpos, fpos, truth, area = process_image_for_roc(img_filename, net, params)
        true_positives += tpos
        false_positives += fpos
        groundtruth += truth
        total_area_km2 += area


    print("========= %d images"%len(img_filenames))
    print("========= %f km^2"%total_area_km2)
    print("========= %d targets"%len(groundtruth))

        
    plot_roc(true_positives, false_positives, groundtruth, total_area_km2)

    print('done')
    
    
    

