# Masked saved in girder item with the image.
# positive, negative, unknown  (g,r,b)

#if debugging, use ["output", "loss","batch"]


import signal
import sys
import shutil
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# The target path is hard coded.
#import fcnn116 as target_last
import skip116 as target
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

    first = True
    count = 0    
    
    for batch in range(params['num_batches']):
        print("== Batch %d"%batch)
        input_np, truth_np, dont_care_np = data.sample_batch(params)

        # Scale to 0->1
        input_np = input_np.astype(np.float32)/255.0
        input_np = np.moveaxis(input_np, 3, 1)
        truth_np = np.array(truth_np)

        input_tensor = torch.from_numpy(input_np).float()
        truth_tensor = torch.from_numpy(truth_np).float()
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda(params['gpu'])
            truth_tensor = truth_tensor.cuda(params['gpu'])

        # Extract the ignore mask from the truth values.  Ignore bit is 128
        ignore_mask_tensor = torch.from_numpy(dont_care_np)
        
        # learning rate change with batch size?
        # create your optimizer
        optimizer = optim.SGD(net.schedule_parameters(), lr=params['rate'])

        # loss function
        criterion = torch.nn.MSELoss(reduce=False)
        #ipdb.set_trace()
        
        for mini in range(params['num_minibatches']):  # loop over the dataset multiple times
            running_loss = 0.0
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output_tensor = net(input_tensor)
            output_tensor = smax(output_tensor)

            tmp_out = output_tensor[:,1,...]
            loss = criterion(tmp_out, truth_tensor)

            if 'debug' in params:
                if mini == params['num_minibatches']-1 or mini == 0:
                    if 'output' in params['debug']:
                        output = output_tensor
                        for idx in range(len(input_np)):
                            tmp = (output[idx,1]).cpu().detach().numpy()
                            cv2.imwrite("debug/d%d_%d_output%d.png"%((batch%2), idx, mini), tmp*255)
            
                    if 'batch' in params['debug']:
                        for idx in range(len(input_np)):
                            tmp = np.moveaxis(input_np[idx], 0,2)*255
                            cv2.imwrite("debug/d%d_%d_input.png"%((batch%2), idx), tmp[...,0:3])
                            cv2.imwrite("debug/d%d_%d_inputP.png"%((batch%2), idx), tmp[...,3])
                            cv2.imwrite("debug/d%d_%d_truth.png"%((batch%2), idx),
                                        (truth_np[idx]*255).astype(np.uint8))

                    #if 'loss' in params['debug']:
                    #    tmp = loss.cpu().detach().numpy()
                    #    for idx in range(len(input_np)):
                    #        cv2.imwrite("debug/d%d_%d_loss1_%d.png"%((batch%2), idx, mini), tmp[idx]*255)

            # Zero out mask pixels.
            loss[ignore_mask_tensor>128] = 0.0

            if 'debug' in params and 'loss' in params['debug']:
                if mini == params['num_minibatches']-1 or mini == 0:
                    tmp = loss.cpu().detach().numpy()
                    for idx in range(len(input_np)):
                        cv2.imwrite("debug/d%d_%d_loss2_%d.png"%((batch%2), idx, mini), tmp[idx]*255)

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
            
        #ipdb.set_trace()

        if running_loss < start_loss:
            #if not params['debug'] or len(params['debug']) == 0:
            if True:
                # Save the weights
                filename = os.path.join(params['folder_path'], params['target_group'],
                                        'model%d.pth'%params['input_level'])
                print("Saving network " + filename)
                if os.path.isfile(filename):
                    shutil.move(filename, os.path.join(params['folder_path'],
                                                       params['target_group'], \
                                                       'model_backup.pth'))
                torch.save(net.state_dict(), filename)
                #schedule = net.schedule_idx
                '''
                if schedule < 12:
                    schedule += 1
                    net.set_schedule(schedule)
                    params['rf_size'] = net.get_rf_size()
                    params['rf_stride'] = net.get_rf_stride()
                    print("schedule = %d"%schedule)
                '''
                

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
    my_utils.reset_batch_norm(net)
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
    chip_size = int(math.ceil(params['input_dim'] * math.sqrt(2)))
    data.sample_negative_chips(image_data, image,  chip_size, net, 100)

    for chip in data.neg_chips:
        chip.save_debug_images("negative")
    



def test_sample_batch(data, net, params):
    image_data = data.get_image_data(0)
    image_data.load_annotations()
    image = image_data.load_image()
    chip_size = int(math.ceil(params['input_dim'] * math.sqrt(2)))
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



def main_train(net, params):
    if params['shock'] > 0.0:
        net_utils.shock_weights(net, params['shock'])
    if torch.cuda.is_available():
        net.cuda(params['gpu'])

    data = d.TrainingData(params)

    #for epoch in range(params['num_epochs']):
    epoch = 0
    while True:
        # load the chips for the first image. (Prime the pump)
        net.eval()

        # Load the next image.
        data.incremental_load()
        if 'debug' in params:
            data.save_chips()

        print("==== Epoch %d"%epoch)
        epoch += 1
        train(net, data, params)
        # dangerous
        os.system('rm %s/debug/*.png'%params['target_group'])
        data.prune_chips()

        params['rate'] *= 0.95
        print("rate %f"%params['rate'])
        data.save_chips()
            

#=================================================================================
def load_net(params):
    net = target.net()
    if 'schedule' in params:
        net.set_schedule(params['schedule'])
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
        #if not params['debug']:
        torch.save(net.state_dict(), filename)

    return net


def exit_gracefully(signum, frame):
    # restore the original signal handler as otherwise evil things will happen
    # in raw_input when CTRL+C is pressed, and our signal handler is not re-entrant (raw_input was renamed to input in python3)
    signal.signal(signal.SIGINT, original_sigint)
    
    try:
        if input("\nReally quit? (y/n)> ").lower().startswith('y'):
            if input("\nSave net? (y/n)> ").lower().startswith('y'):
                global net, params
                filename = os.path.join(params['folder_path'], params['target_group'],
                                        'model%d.pth'%params['input_level'])
                print("Saving network " + filename)
                torch.save(net.state_dict(), filename)

            sys.exit(1)
            
    except KeyboardInterrupt:
        print("  Ok ok, quitting, chill -_-")
        sys.exit(1)

    # This path would restart training.
    # restore the exit gracefully handler here
    signal.signal(signal.SIGINT, exit_gracefully)




# 3 ok,  2 is not
if __name__ == '__main__':
    with open('params.json') as json_file:
        params = json.load(json_file)    

    if 'random_seed' in params:
        random.seed(params['random_seed'])
        np.random.seed(params['random_seed'])
        torch.manual_seed(params['random_seed'])

    net = load_net(params)
    net.train()
    
    # store the original SIGINT handler
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, exit_gracefully)

    main_train(net, params)

