#!/home/local/KHQ/charles.law/envs/pytorch/bin/python
# sbatch -c 3 -p vigilant --gres=gpu:1 main116.py



# Trying to extend adversarial modification to all input images.





# Failed on these:
# ./train/Moscow/3857_9_309_159_20170202_31e1ec5c-62ba-4fbe-9a1f-b7c0a1dc908b.png
# 1.2 GB:  ./train/Denver/3857_11_428_776_20170212_c0201363-a11e-43f6-8554-14db84707f56.png


# Scalable training.
# Load images incrementally.  Load them one at a time, free them immediatly.
# Load larger "chips" around targets and false positive regions.
# I select negative chips by first creating an error map of the whole image.
#   I compute a target number of samples by using the total number of positive chips
#   and the realtive magnitude of the error map. (huristic)
# Keep track of the chip error.
# sample batch: select chips with probability based on error,
#   augments the chip with rotation, scale, then crop smaller images for training





# Load images one at a time. Never have more than one full image in memory at a time.
# Strategy:
# - Cache negative and positive image chips.
# - Positive chips are static.
# - Negative chips get sampled at some interval.  Negative chips get pruned at the same time.
# - Chips have their own sample probabliltiy.
# - Chips are big enough to have rotational augmentatnion.
#   - This is necessary because positive chips are static.
# - translation augmentation is accomplished by simply passing larger images through the network.
# - Each chip has its own truth mask. each chip has its own pdf.



# finished:
# - Test read file names and create empty image_data objects.
# - Test Load positive chips for a single image.
# - Test generation of truth masks (positve)
# - Sample negative chips for a single image.
# - Write code to sample batch images and truth from the chips.
# - Subsample large targets.



# TODO:
# Add adversarial



# Old ...........  Drop?
# - figure out why max of negative error map is always 0.1
# - Save out histograms of error for positive and negative chips.
# - Abstract the ground truth (implicit annotations, explicit mask).



import sys
import shutil
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# The target path is hard coded.
import fcnn116 as target_last
import fcnn as target
import pdb
import ipdb
import scipy.misc
import csv
import random
import time
import cv2
import math
import os
import numpy as np
from pprint import pprint
import my_utils
import girder as g
# todo: get rid of the misc function references.
import data_mask as d
import net_utils
#from adversarial import *
import json



# Just generate the output map for the all the loaded iamges.
# also save out the truth images for comparison.
# only one grb image, specified by "debug_channels," for now.
def test_images(net, data, params):
    dir_path = os.path.join(params['folder_path'], "train", params['target_group'], 'test')
    ensuredir(dir_path)

    for idx in range(data.get_number_of_images()):
        image_data = data.get_image_data(idx)
        truth_np = image_data['truth']
        truth_img = truth_np[:, :, params['debug_channels']]

        image_np = image_data['image']
        output_np = execute_large_image(net, image_np, params)
        
        output_img = output_np[:, :, params['debug_channels']]
        tmp = np.max(output_img)
        tmp = 1
        output_img = output_img * (255/tmp)
        output_img = np.clip(output_img, 0, 255)
        np.clip(output_img, 0, 255, out=output_img)
        output_img = output_img.astype(np.uint8)
        
        file_root = image_data['file_root']
        file_path = os.path.join(dir_path, "%s_out.png"%file_root)
        cv2.imwrite(file_path, output_img)
        file_path = os.path.join(dir_path, "%s_truth.png"%file_root)
        cv2.imwrite(file_path, truth_img)



def save_debug_training_images(input_np, truth_np, chips, params):
    num = input_np.shape[0]
    for idx in range(num):
        chip = chips[idx]
        file_root = os.path.split(chip.image_data.file_root)[1]
        error = chip.error
        error = int(error*9999)
        image = input_np[idx, ...]
        image = np.moveaxis(image, 0, 2)
        image = image * 255
        image = np.clip(image, 0, 255)
        image = np.array(image).astype(np.uint8)
        file_name = "%s/debug/b_%04d_%s_%d_%d.png"%(params['target_group'], error, file_root, \
                                                    chip.center[0], chip.center[1])
        cv2.imwrite(file_name, image)
        truth = truth_np[idx, ...]
        truth[truth == 1] = 255
        truth = truth.astype(np.uint8)
        file_name = "%s/debug/b_%04d_%s_%d_%d_t.png"%(params['target_group'], error, file_root, \
                                                      chip.center[0], chip.center[1])
        cv2.imwrite(file_name, truth)
        

    
    
#=================================================================================
# Train to create segmentation mask for airplanes in pytorch
def train(net, data, params):
    num_adversarial_images = 0 # 2
    smax = nn.Softmax(dim=1)

    for batch in range(params['num_batches']):
        print("== Batch %d"%batch)
        input_np, truth_np, dont_care_np = data.sample_batch(params)

        #cv2.imwrite("input0.png", input_np[0][...,0:3])
        #cv2.imwrite("inputP0.png", input_np[0][...,3])
        #cv2.imwrite("input1.png", input_np[1][...,0:3])
        #cv2.imwrite("inputP1.png", input_np[1][...,3])
        
        # Scale to 0->1
        input_np = input_np.astype(np.float32)/255.0
        input_np = np.moveaxis(input_np, 3, 1)
        truth_np = np.array(truth_np).astype(np.long)

        input_tensor = torch.from_numpy(input_np).float()
        truth_tensor = torch.from_numpy(truth_np).long()
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda(params['gpu'])
            truth_tensor = truth_tensor.cuda(params['gpu'])

        # Extract the ignore mask from the truth values.  Ignore bit is 128
        dont_care_np = (dont_care_np > 128).astype(np.int)
        ignore_mask_tensor = torch.from_numpy(dont_care_np)
        
        # learning rate change with batch size?
        # create your optimizer
        optimizer = optim.SGD(net.parameters(), lr=params['rate'])

        # loss function
        criterion = torch.nn.CrossEntropyLoss(reduce=False)
    
        for mini in range(params['num_minibatches']):  # loop over the dataset multiple times
            running_loss = 0.0
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output_tensor = net(input_tensor)

            # not needed with only one target
            # Ignore all but the targeted indexes for the loss function.
            #tmp_out = output_tensor[:,params['target_indexes'],...]

            tmp_out = output_tensor
            loss = criterion(tmp_out, truth_tensor)
            # Zero out mask pixels.
            loss[ignore_mask_tensor] = 0.0
            loss_scalar = loss.mean()
            
            loss_scalar.backward()  #loss.backward(retain_graph=True)
            optimizer.step()
            
            # print statistics
            running_loss = loss_scalar.item() * 100
            if mini == 0:
                # Remove the loss generated by adversarial images
                loss_np = loss.detach().cpu().numpy()
                num = len(loss_np)-num_adversarial_images
                data.record_error(loss_np[0:num, ...], params['heatmap_decay'])
                start_loss = running_loss
                print(" %d: loss: %.3f" % (mini + 1, running_loss))
                print("")
            else:
                if running_loss > last_loss:
                    # Turn on rend and then turn it off
                    print("\033[0;31m\033[1A %d: loss: %.3f" % (mini + 1, running_loss))
                    print("\033[0m")
                else:
                    print("\033[1A %d: loss: %.3f" % (mini + 1, running_loss))
                #print("%d: loss: %.3f" % (mini + 1, running_loss))
            last_loss = running_loss
                
        if running_loss < start_loss:
            # Save the weights
            filename = os.path.join(params['folder_path'], params['target_group'], 'model%d.pth'%params['input_level'])
            print("Saving network " + filename)
            if os.path.isfile(filename):
                shutil.move(filename, os.path.join(params['folder_path'], params['target_group'], \
                                                   'model_backup.pth'))
            if not params['debug']:
                torch.save(net.state_dict(), filename)



def save_debug_input(input_tensor, root_name):
    print(" ---- saving %s"%root_name)
    input_np = input_tensor.detach().cpu().numpy()
    for idx in range(len(input_np)):
        img = input_np[idx]
        img = np.moveaxis(img, 0, 2)
        peak = np.max(img)
        #print(peak)
        img = img * (255.0 / peak)
        filename = "debug/%s_%d.png"%(root_name, idx)
        cv2.imwrite(filename, img.astype(np.uint8))


        

def adversarial(net, input_np, truth_np, params):
    """ Modify the inputs to be more 'false'.
    """
    input_tensor = torch.from_numpy(input_np).float()
    truth_tensor = torch.from_numpy(truth_np).long()

    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda(params['gpu'])
        truth_tensor = truth_tensor.cuda(params['gpu'])

    input_tensor.requires_grad = True

    # Extract the ignore mask from the truth values.  Ignore bit is 128
    ignore_mask_tensor = (truth_tensor > 127)
    truth_tensor[ignore_mask_tensor] = 0
            
    # learning rate change with batch size?
    # create your optimizer, but change the input instead of the weighs.
    #optimizer = optim.SGD([input_tensor], lr=params['rate'])
    #optimizer = torch.optim.Adam([input_tensor])
    # loss function
    criterion = torch.nn.CrossEntropyLoss(reduce=False)
    relu = nn.ReLU()

    save_debug_input(input_tensor, "start")    
    for mini in range(params['num_minibatches']):  # loop over the dataset multiple times
        # stop propagation of gradient through interations.
        input_tensor = Variable(input_tensor, requires_grad=True)

        running_loss = 0.0
        # zero the parameter gradients
        #optimizer.zero_grad()

        # forward + backward + optimize
        output_tensor = net(input_tensor)

        # not needed with only one target
        # Ignore all but the targeted indexes for the loss function.
        #tmp_out = output_tensor[:,params['target_indexes'],...]

        tmp_out = output_tensor
        loss = criterion(tmp_out, truth_tensor)
        # Zero out mask pixels.
        loss[ignore_mask_tensor] = 0.0

        # The negative should cause the optimizer to maximize loss.
        loss = -loss.mean()
        print(loss.item())
        loss.backward()  


        #optimizer.step()
        grad = input_tensor.grad
        gm = grad.max()
        im = input_tensor.max()
        # Hack to pick a learning rate.
        # This turns out to be quite high, but it works.
        rate = im / (30*gm)
        input_tensor = input_tensor - (grad*rate)
        # Keep the scale of the noise image from 0 to 1
        t1 = input_tensor.max()
        if t1 > 1.0:
            input_tensor = input_tensor / t1
        input_tensor = relu(input_tensor)

        

    save_debug_input(input_tensor, "end")    
                


# Test the generation of adversarial input.
def test_noise(params):
    random.seed(500)

    # Load the network model
    print('loading net')
    net = load_net(params)    
    if torch.cuda.is_available():
        net.cuda(params['gpu'])

    num_images = 16
    min_scale = 0.8
    girder_data_folder_ids = ["58955d203f24e50b2776aaff"]
    gc = g.get_gc()
    # Load the training data
    arrow_sets = g.load_arrows_and_wings(girder_data_folder_ids, min_scale, gc)
    _,truth_np,arrows = sample_arrows(arrow_sets, num_images, \
                                      min_scale, params['rf_size'])
    
    input_tensor, output_tensor = get_adversarial_tensors(net, truth_np, 2000)

    images = input_tensor.cpu().numpy()
    images = np.moveaxis(images, 1, 3)*255.0 
    images = np.clip(images, 0, 255)
    images = np.array(images, dtype=np.uint8)
    
    print("Saving adversiarial images")
    in_np = input_tensor.cpu().numpy()
    out_np = output_tensor.cpu().numpy()
    upload_debug_images(in_np, [out_np], "wing5_noise", arrows=arrows)



def main_test_images(params):
    gc = g.get_gc()

    # Load the network model
    print('loading net')
    net = load_net(params)    
    #shock_weights(net)
    if torch.cuda.is_available():
        net.cuda(params['gpu'])
    # Lock batch normalization
    net.eval()
    
    # Load the training data
    data = data_json.load_data(params) # num=600)

    test_images(net, data, params)


def main_show_pdfs(params):
    data = data_json.load_data(params)
    data.show_pdfs()




    


def test_execute_large_image(data, net):
    image_data = data.get_image_data(0)
    image = data.load_image(image_data)
    out = net_utils.execute_large_image(net, image, params)

    out = out * 255
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    cv2.imwrite('debug.png', out)

    
        

def test_load_negative_chips(data, net, params):
    image_data = data.get_image_data(0)
    image_data.load_annotations()
    chip_size = int(math.ceil(params['input_size'] * math.sqrt(2)))
    data.sample_negative_chips(image_data, image,  chip_size, net, 100)

    for chip in data.neg_chips:
        chip.save_debug_images("negative")
    



def test_sample_batch(data, net, params):
    image_data = data.get_image_data(0)
    image_data.load_annotations()
    image = image_data.load_image()
    chip_size = int(math.ceil(params['input_size'] * math.sqrt(2)))
    image_data.load_positive_chips(image, chip_size, params)

    for chip in data.pos_chips:
        chip.save_debug_images("positive")

    data.sample_negative_chips(image_data, image, chip_size, net, 100)

    for chip in data.neg_chips:
        chip.save_debug_images("negative")
    
    images, truths = data.sample_batch(params)

    for idx in range(len(images)):
        image = images[idx]
        cv2.imwrite("batch_%d_image.png"%idx, image)
        truth = truths[idx]
        truth[truth==1] = 255
        cv2.imwrite("batch_%d_truth.png"%idx, truth)



def main_train(params):
    net = load_net(params)
    if torch.cuda.is_available():
        net.cuda(params['gpu'])

    # A hacky way to train up through the levels.
    net.set_schedule(params['schedule'])
    params['rf_size'] = net.get_rf_size()
    params['rf_stride'] = net.get_rf_stride()
    
    


    data = d.TrainingData(params)

    #for epoch in range(params['num_epochs']):
    epoch = 0
    while True:
        # load the chips for the first image. (Prime the pump)
        net.eval()

        # Load the next image.
        data.incremental_load()
        if params['debug']:
            data.save_chips()
        
        print("==== Epoch %d"%epoch)
        epoch += 1
        train(net, data, params)
        # dangerous
        os.system('rm %s/debug/*.png'%params['target_group'])
        data.prune_chips()

        # A hacky way to train up through the levels.
        #if epoch >= 20 and schedule_idx < len(net.schedule):
        #    epoch = 0
        #    schedule_idx += 1
        #    net.set_schedule(schedule_idx)
        #    params['rf_size'] = net.get_rf_size()
        #    params['rf_stride'] = net.get_rf_stride()
        #    print("======= Moving to schedule %d, rf size = %d"%(schedule_idx, \
        #                                                         net.get_rf_size()))
        #    # Keep the chip errors, even though the are not completely valid anymore.
        #    # The truth however is not the same shape. We have to recompute truth images.
        #    data.recompute_chip_truth()

        if epoch%5 == 0:
            params['rate'] *= 0.95
            print("rate %f"%params['rate'])
            data.save_chips()
            

#=================================================================================
def load_net(params):
    net = target.net()
    params['rf_size'] = net.get_rf_size()
    params['rf_stride'] = net.get_rf_stride()
    
    filename = os.path.join(params['folder_path'], params['target_group'], 'model%d.pth'%params['input_level'])
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
        if not params['debug']:
            torch.save(net.state_dict(), filename)

    return net



    
if __name__ == '__main__':
    params = {}
    
    '''
    # We get these from the net now
    #params['rf_stride'] = 4
    #params['rf_size'] = 116
    params['min_augmentation_scale'] = 0.6
    params['data_path'] = '../DigitalGlobe/images'  # ancestor directory for png files.
    params['folder_path'] = '.'  # this is the path to store incremental results.
    params['truth_radius'] = 30
    params['ignore_radius'] = 60
    params['gpu'] = 0 #3 # 0
    params['num_epochs'] = 100
    # Batches / Epoch: Load a new image every # batchs
    params['num_batches'] = 30
    # resample batch training images every # cycles
    params['num_minibatches'] = 8 #20
    params['rate'] = 0.005
    params['heatmap_decay'] = 0.2
    params['debug'] = False
    params['target_group'] = 'fcnn116'
    params['max_num_training_images'] = 5000
    # impacts gpu memory usage
    params['input_size'] = 116
    params['batch_size'] = 64

    params['image_cache_dir'] = '../cached_images'
    params['chip_cache_dir'] = '../cached_chips'
    params['input_level'] = 3
    params['schedule'] = 4

    with open('params.json', 'w') as outfile: json.dump(params, outfile)
    '''
    with open('params.json') as json_file: params = json.load(json_file)


    main_train(params)
    sys.exit()



    print('done')
    
    
    

