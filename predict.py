
import math
import torch
# The target path is hard coded.
import skip116 as target
import cv2
import os
import numpy as np
import girder as g
import net_utils
import ipdb
import json
try:
    import matplotlib.pyplot as plt
except:
    plt = None
    print("no plotting")
    
            

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


    #this is for when the image is too large to process all at once
    name = 'gwenda.law'
    
    annotation_id = g.get_annotation_id_from_name(item_id, name, gc)
    
    center, width, height = g.get_annotation_loc_from_id(annotation_id, gc)
    
    spacing = math.pow(2,params['input_level'])
    
    x = center[0]
    y = center[1]
    w = int(width/spacing)
    h = int(height/spacing)
    
    #image = g.get_image_cutout(gc, item_id, (x,y), w, h, scale=1.0/spacing, cache='cache')
    

    image = g.get_image(item_id,level=params['input_level'])
    masks = g.get_image_file(gc,item_id,'masks.png')
    #masks = masks[int(y/masks.shape[1]-h/masks.shape[1]):int(y/masks.shape[1]+h/masks.shape[1]),int(x/masks.shape[0]-w/masks.shape[0]):int(x/masks.shape[0]+w/masks.shape[0])]

    error_map = g.get_image_file(gc,item_id,'error_map%d.png'%params['input_level'])
    #if error_map is not None:
        #error_map = error_map[int(y/error_map.shape[1]-h/error_map.shape[1]):int(y/error_map.shape[1]+h/error_map.shape[1]),int(x/error_map.shape[0]-w/error_map.shape[0]):int(x/error_map.shape[0]+w/error_map.shape[0])]
    if error_map is None and not(masks is None):
        error_map = np.zeros(masks.shape)
        
    net = load_net(params)
    if torch.cuda.is_available():
        net.cuda(params['gpu'])

    # This is automatic,  I am manually controlling the prediction from the last
    # level in params.json, so I can test with and without it.
    #prediction_level = params['input_level']+1
    #prediction = g.get_image_file(gc,item_id,'prediction%d.png'%prediction_level)
    prediction = g.get_image_file(gc,item_id,params['prediction_heatmap'])

    if prediction is None:
        net_in = np.dstack((image, np.zeros(image.shape[:-1])))
    else:
        if len(prediction.shape) > 1:
            prediction = prediction[...,1]
        # Make sure opencv is not swapping using x for y.
        dx = image.shape[1]
        dy = image.shape[0]
        prediction = cv2.resize(prediction, (dx,dy), interpolation=cv2.INTER_AREA)
        # Add the prediction from the previous level as 4 channel to the input.
        net_in = np.dstack((image, prediction))
    

    net_out = net_utils.execute_large_image(net, net_in, params)

    net_out *= 255.999
    net_out = np.clip(net_out, 0, 255)

    net_out = net_out.astype(np.uint8)
    net_out_flip = net_out[:,:,0]
    net_out = net_out[:,:,1]
    net_predict = np.dstack((net_out, net_out, np.zeros(net_out.shape), net_out))
    
    cv2.imwrite('prediction%d.png'%params['input_level'],net_predict)
    heatmap  = g.Heatmap(net_predict)
    heatmap.save_to_girder(item_id, 'prediction%d.png'%params['input_level'])
    
    # Update pdf / error map
    if not(masks is None):
        dx = masks.shape[1]
        dy = masks.shape[0]
        net_out = cv2.resize(net_out, (dx, dy))
        net_out_flip = cv2.resize(net_out_flip, (dx, dy))
        
        positive = masks[:,:,0]
        negative = 255-masks[:,:,0]
        alpha = masks[:,:,3]
        
        positive_error_map = net_out_flip
        positive_error_map[positive<128] = 0
        positive_error_map[alpha<128] = 0
        
        negative_error_map = net_out
        negative_error_map[negative<128] = 0
        negative_error_map[alpha<128] = 0
    
        # Save error map / pdf back to girder.
        #positive_error_map = positive_error_map.astype(np.uint8)
        #negative_error_map = negative_error_map.astype(np.uint8)
        error_map = np.dstack((np.zeros(positive_error_map.shape),
                               positive_error_map,negative_error_map))
        # Should I add an alpha just for display?
        
        cv2.imwrite('error_map%d.png'%params['input_level'],error_map)
        heatmap  = g.Heatmap(error_map)
        heatmap.save_to_girder(item_id, 'error%d.png'%params['input_level'])
    

