


import torch
# The target path is hard coded.
import fcnn as target
import cv2
import os
import numpy as np
import girder as g
import net_utils
import ipdb
import matplotlib.pyplot as plt
import json

    
            

#=================================================================================
def load_net(params):
    net = target.net()
    net.set_schedule(params['schedule'])
    params['rf_size'] = net.get_rf_size()
    params['rf_stride'] = net.get_rf_stride()
    net.eval()
    
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
        torch.save(net.state_dict(), filename)

    return net


    
        
if __name__ == '__main__':
    params = {}
    with open('params.json') as json_file: params = json.load(json_file)

    gc = g.get_gc()
    
    #item_id = '5915d969dd98b578723a09c2' #c
    #item_id = '5915d9c1dd98b578723a09c5' #d
    #item_id = '5915da13dd98b578723a09c8' #e
    item_id = '5915da6add98b578723a09cb' #f

    image = g.get_image(item_id,level=params['input_level'])
    masks = g.get_image_file(gc,item_id,'masks.png')
    error_map = g.get_image_file(gc,item_id,'error_map%d.png'%params['input_level'])
    if error_map is None and not(masks is None):
        error_map = np.zeros(masks.shape)
        
    net = load_net(params)
    if torch.cuda.is_available():
        net.cuda(params['gpu'])
    
    prediction_level = params['input_level']+3
    prediction = g.get_image_file(gc,item_id,'prediction%d.png'%prediction_level)
    if prediction is None:
        image = np.dstack((image, np.zeros(image.shape[:-1])))
    else:
        if len(prediction.shape) == 3:
            prediction = prediction[...,0]
            prediction = cv2.resize(prediction,(image.shape[1],image.shape[0]), interpolation=cv2.INTER_AREA)

            image = np.dstack((image, prediction))


    net_out = net_utils.execute_large_image(net,image,params)

    net_out *= 255.999
    net_out = np.clip(net_out,0,255)
    
    net_out = net_out.astype(np.uint8)
    net_out_flip = net_out[:,:,0]
    net_out = net_out[:,:,1]
    
    plt.figure(figsize=(8,10))

    
    cv2.imwrite('prediction%d.png'%params['input_level'],net_out)
    files = gc.listFile(item_id)
    for f in files:
        if f['name'] == 'prediction%d.png'%params['input_level']:
            gc.delete('file/%s'%f['_id'])
        if f['name'] == 'error_map%d.png'%params['input_level']:
            gc.delete('file/%s'%f['_id'])
    gc.uploadFileToItem(item_id, 'prediction%d.png'%params['input_level'])
    
    # Update pdf / error map
    if not(masks is None):

        net_out = cv2.resize(net_out,(masks.shape[1],masks.shape[0]))
        net_out_flip = cv2.resize(net_out_flip,(masks.shape[1],masks.shape[0]))
    
        
        unknown = masks[:,:,0]
        positive = masks[:,:,1]
        negative = masks[:,:,2]
        
        
        positive_error_map = net_out_flip
        positive_error_map[positive<128] = 0
        
        negative_error_map = net_out
        negative_error_map[negative<128] = 0
        
    
        # Save error map / pdf back to girder.
        positive_error_map = positive_error_map.astype(np.uint8)
    
        
        negative_error_map = negative_error_map.astype(np.uint8)
    
        
        
        error_map = np.dstack((np.zeros(positive_error_map.shape),positive_error_map,negative_error_map))
        
        #error_map[0:1150,:] = 0
        #error_map[1550:,:] = 0
        #error_map[:,0:500] = 0
        #error_map[:,1010:] = 0
        
        cv2.imwrite('error_map%d.png'%params['input_level'],error_map)
        gc.uploadFileToItem(item_id, 'error_map%d.png'%params['input_level'])
    

