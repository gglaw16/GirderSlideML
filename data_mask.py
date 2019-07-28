# api:
# - data = TrainingData(params)#   returns a data object.
# - data.incremental_load()
# api:
# - data = TrainingData(params)#   returns a data object.
# - data.incremental_load()
#   Load the next image and get positive and negative chips.
# - data.prune_chips()
#   remove low error chips to limit the number of cached chips.
# - data.save_chips()
#   saves current chips for restart.
# - input_np, truth_np = data.sample_batch()
#   creates batch numpy arrays for training. 
# - record_error(loss_np, decay)
#   set new error for the last sample_batch chips.
# parameters: input_level


# Branched from DigitalGlobe/data_json.py
# Taylored to masks.
#
# Reads a mask from a "heatmap annotation" (expects "user['file_id']")
# TODO: take a sub reagion of this mask to train on.
#
# Writes a prediction to a second heatmap.
# this can start as simply a rectangle annotation.



import sys
import ipdb
import scipy.misc
import random
import csv
import cv2
import math
import json
import os
import glob
import shutil
import numpy as np
from pprint import pprint
from my_utils import *
import net_utils
import girder as g
import pylaw
try:
    import matplotlib.pyplot as plt
except:
    print("no plotting")
    
# ============= external
# load_data(params)


# ============= internal



def pylaw2(pdf, radius, sample_count):
    """
    An alternative to pylaw.sample
    Find peaks.  Keep neighboring peaks a minimim distance away.
    greedy algorithm
    returns (points, scores)
    """
    points = np.zeros((sample_count, 2))
    # Also return the response
    scores = np.zeros((sample_count))
    for idx in range(sample_count):
        (y,x) = np.unravel_index(np.argmax(pdf, axis=None), pdf.shape)
        points[idx] = (y,x)
        scores[idx] = pdf[y,x]
        ymin = max(y-radius, 0)
        ymax = min(y+radius, pdf.shape[0])
        xmin = max(x-radius,0)
        xmax = min(x+radius, pdf.shape[1])
        pdf[ymin:ymax, xmin:xmax] = 0
    return points, scores




# Utility function that transforms and crops both an image and mask.
# I think image_center is (x, y) but I should verify this....
# Internal function to crop a chip from an image.
# It works on input images and output ground truth.
# It also handles augmentation.
# This does not work on grayscale (1 channel) yet.
# Keep it simple.  Make the caller handle truth and input in separate calls.
# returns the sampled image.
def sample_image(image, image_center, sample_size, rotation=None, scale=1.0, mirror=True,
                 pad_value=0, interpolation=cv2.INTER_LINEAR):
    #""
    #image: np array of shape (dimy, dimx, channels)
    #point:  (y,x) center of the sampled chip.
    #rotation: counter clockwise in degrees
    #scale: 2=> twice as big
    #pad_value:  All chip pixels outside image bounds get set to this value.
    #returns image_chip
    #""
    if len(image.shape) == 2:
        image = image.reshape(image.shape + (1,))
    height,width,num_channels = image.shape

    x = image_center[1]
    y = image_center[0]

    sample_center = (sample_size/2, sample_size/2)    
    offset = [image_center[1] - sample_center[1], image_center[0] - sample_center[0]]

    if rotation is None:
        rotation = 360.0 * random.random()
    
    # rotate
    rows = height
    cols = width
    M = cv2.getRotationMatrix2D((x,y), rotation, scale)
    # add a reflection
    if mirror:
        M[1][0] = -M[1][0]
        M[1][1] = -M[1][1]
        M[1][2] = 2*y - M[1][2]
    # Shift to put the 'point' in the center of the sample image.
    M[0][2] -= offset[0];
    M[1][2] -= offset[1];
    # add a reflection
    #if mirror:
    #    M[1][0] = -M[1][0]
    #    M[1][1] = -M[1][1]
    #    M[1][2] = sample_size - M[1][2]

    # Pad does not accept tuples over length 4.
    if pad_value == 0:
        chip = cv2.warpAffine(image,M,(sample_size, sample_size), flags=interpolation)
    else:
        pad_vector = tuple([pad_value]*num_channels)
        chip = cv2.warpAffine(image,M,(sample_size, sample_size), \
                              borderMode=cv2.BORDER_CONSTANT, \
                              borderValue=pad_vector, flags=interpolation)
        #  flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS

    return chip





class ChipData:
    """
    Image chip (big enough for augmentation).
    (also contains truth map)
    json is for restart chips.
    """
    
    # I am not sure about storing the image data here.  The image data references the chip.
    def __init__(self, chip=None, truth=None, center=None, image_data=None, spacing=1):
        # We need scale (1/spacing) to compute the mask from annotation.
        self.spacing = spacing
        self.image_data = image_data
        self.file_root = None
        self.json_file_path = None
        # Note:  This center is in input coordinates
        self.center = center
        self.chip = chip
        self.truth = truth
        self.error = 1.0
        
        
    def get_file_root(self):
        if self.image_data:
            return self.image_data.file_root
        elif self.file_root:
            return self.file_root
        return ""


    def save(self, dirname, tag):
        """
        Saves a file for real loading / restart.
        json, with two pngs for image and truth.
        Tag string is not required internally. It is for the owner to create
        chip sets.
        """

        # create Json meta data file.
        meta = {'spacing': self.spacing,
                'center': self.center,
                'error': str(self.error),
                'item_id': self.image_data.item_id,
                'tag': tag}

        # Get unique filename
        error = "%02d"%(self.error*99)
        id = random.randint(10000,99999)
        filepath = os.path.join(dirname, tag, 'e%s_%d.json'%(error,id))
        while os.path.isfile(filepath):
            id += 1
            filepath = os.path.join(dirname, tag, 'e%s_%d.json'%(error,id))
        # save out the input and truth images.
        img_filepath = os.path.join(dirname, tag, 'e%s_%d_img_a.png'%(error,id))
        cv2.imwrite(img_filepath, self.chip[:,:,0:3])
        img_filepath = os.path.join(dirname, tag, 'e%s_%d_img_b.png'%(error,id))
        cv2.imwrite(img_filepath, self.chip[:,:,3])
        meta['chip'] = img_filepath
        
        if (not self.truth is None):
            truth_filepath = os.path.join(dirname, tag, 'e%s_%d_truth_a.png'%(error,id))
            cv2.imwrite(truth_filepath, self.truth[:,:,0:3])
            truth_filepath = os.path.join(dirname, tag, 'e%s_%d_truth_b.png'%(error,id))
            cv2.imwrite(truth_filepath, self.truth[:,:,3])
            meta['truth'] = truth_filepath

        with open(filepath, 'w') as fp:
            json.dump(meta, fp)

        
    def load(self, filepath, view_data, params):
        """
        Loads a chip and all its info back in.
        I do not bother setting the image_data (although I have info to do that)
        """
        if not os.path.exists(filepath):
            return False
        
        with open(filepath, 'r') as fp:
            meta = json.load(fp)

        self.spacing = meta['spacing']
        self.center = meta['center']
        if 'score' in meta:
            self.error = float(meta['score'])
        if 'error' in meta:
            self.error = float(meta['error'])
        self.chip = cv2.imread(meta['chip'])

        if  'truth' in meta:            
            self.truth = cv2.imread(meta['truth'], 0)
            
        self.file_root = meta['image_data_root']
        root = os.path.split(self.file_root)[1]
        self.image_data = view_data.get_image_data_from_root(root)

        if 'json_file_path' in meta:
            self.json_file_path = meta['json_file_path']

        if self.truth is None or self.chip is None:
            return False
        return True
        

    def augment(self, input_dim, truth_dim, params):
        """
        Chooses a random rotation, intensity and constrast transformation.
        Returns input and truth images cropped to the appropriate size for training.
        There is no translational augmentation because convolution takes care of that.
        """
        # augmentation
        rotation = random.random() * 360.0
        mirror = random.random() > 0.5
        # scale the ground truth to fit in the rf.
        
        
        scale = 1
        brightness = random.random()*30.0 - 25.0
        contrast = random.random()*0.4 + 0.8

        image_center = self.chip.shape[0] / 2.0
        image_center = (image_center, image_center)

        truth_center = self.truth.shape[0] / 2.0
        truth_center = (truth_center, truth_center)

        image = sample_image(self.chip, image_center, input_dim, rotation=rotation, \
                             mirror=mirror, scale=scale, pad_value=180)

        # handle brightness contrast augmentaiton
        # TODO: Fix this so it does not change the prediction channel.
        #image = (image*contrast)+brightness;
        np.clip(image, 0, 255, out=image)
        image = np.uint8(image)

        truth = sample_image(self.truth, truth_center,
                             truth_dim, rotation=rotation, \
                             mirror=mirror, scale=scale, pad_value=0, \
                             interpolation=cv2.INTER_NEAREST)
        ignore = np.invert(truth[...,3])

        # at some stage, pixles are interpolated 
        truth[truth>128] = 255
        truth[truth<128] = 0
        ignore[ignore>128] = 255
        ignore[ignore<128] = 0
        
        return image, truth[...,1].astype(np.float)/255.0, ignore

    
    def save_debug_images(self, dir_path):
        s = int(self.error*9999)
        x = int(self.center[0])
        y = int(self.center[1])
        root = self.image_data.item_id
        file_path = os.path.join(dir_path, "s%04d_x%d_y%d_%s_chip.png"%(s, x, y, root))
        cv2.imwrite(file_path, self.chip)
        if not self.truth is None:
            file_path = os.path.join(dir_path, "s%04d_x%d_y%d_%s_truth.png"%(s, x, y, root))
            tmp = self.truth.copy()
            tmp[tmp == 1] = 255
            cv2.imwrite(file_path, tmp)


class ImageData:
    """
    Holds all the training data and info associated with one girder image.
    """
    
    def __init__(self, item_id=None, params=None):
        self.params = params
        self.item_id = item_id
        self.annotation = {}
        # get the dimensions of the wholeimage
        gc = g.get_gc()
        resp = gc.get('item/%s/tiles'%item_id)
        self.x_dim = resp['sizeX']
        self.y_dim = resp['sizeY']
        
        
    def get_item_id(self):
        return self.item_id

    # Does not keep a reference to the image in this object.
    def load_image(self):
        return g.get_image(self.item_id, cache=self.params['image_cache_dir'])

    # not used.
    def get_negative_error_image(self, image, net):
        net_out = net_utils.execute_large_image(net, image, self.params)

        # Get the negative channel.
        # Invert it to compute error from activation.
        error = 1.0 - net_out[..., 0]

        # Threshold: Sort of like margin/hinge
        error = error - 0.4
        # I am not sure I like chopping off the peaks.
        # we sample the locations with max error.
        error = np.clip(error, 0.0, 0.7)

        truth_spacing = net.get_rf_stride()
        
        # set all the positive regions to 0 (should we do a gaussian window?)
        # This error map is computed at "base resolution" (some chips are subsampled)
        truth_spacing = net.get_rf_stride()
        for e in self.annotations['aircraft']:
            # image pixel coordinate system.
            outer_radius = 2 * e['radius'] * self.params['truth_radius'] / self.params['rf_size']
            # output coordinate system
            outer_radius = outer_radius / truth_spacing            
            # convert center from input to output coordinate system.
            x = (e['center'][0] / truth_spacing)
            y = (e['center'][1] / truth_spacing)
            cv2.circle(error, (int(x),int(y)), int(outer_radius), (0), thickness = -1)

        return error

    # I suspect this is not working right.  I am getting a shift
    def crop_error_map(self, error_map, truth_map, margin):
        """
        Crop the error map to make sure that truth chips are always in the truth map.
        error_map: {'image': img, 'region': [x0,y0,x1,y1]}
        truth_map: {'image': img, 'region': [x0,y0,x1,y1]}
        region and margin are in level0 slide coords.
        A new smaller error map is returned
        """
        # Compute the region in level0 slide coordinates
        region = truth_map.region
        region = [region[0] + margin, region[1] + margin,
                  region[2] - margin, region[3] - margin]
        # crop with the error region
        e_region = error_map.region
        region = [max(region[0], e_region[0]), max(region[1], e_region[1]),
                  min(region[2], e_region[2]), min(region[3], e_region[3])]

        # convert region to error pixel coordinates so we can crop.        
        e_spacing_x = (e_region[2] - e_region[0]) / error_map.image.shape[1]
        e_spacing_y = (e_region[3] - e_region[1]) / error_map.image.shape[0]
        min_x = int((region[0]-e_region[0]) / e_spacing_x)
        min_y = int((region[1]-e_region[1]) / e_spacing_y)
        max_x = int((region[2]-e_region[0]) / e_spacing_x)
        max_y = int((region[3]-e_region[1]) / e_spacing_y)

        # Just to be safe
        min_x = max(min_x, 0)
        min_y = max(min_y, 0)
        max_x = min(max_x, error_map.image.shape[1])
        max_y = min(max_y, error_map.image.shape[0])

        cropped_image = error_map.image[min_y:max_y, min_x:max_y]

        cropped_map = g.Heatmap()
        cropped_map.image = cropped_image;
        cropped_map.region = region
        
        return cropped_map


    def sample_chips(self, sample_count, error_map, truth_map,
                     prediction_map=None, debug_tag=None):
        """
        error_map: sample distribution (can be any spacing).
        truth_map: must have same pixel size as output.
        prediction: must have same pixel size as input.
        sample_count:  the number of chips to return.
        Returns a list of high error chips.
        """
        gc = g.get_gc()

        # Compute the dimension of the input and truth chip we will need.
        # Augmentation shrinks the chip so we have to save a larger orginal image here.
        # We also have to leave enough padding for rotation augmentation
        # (without clipping corners).
        min_scale = self.params['min_augmentation_scale']
        in_chip_dim = int(math.ceil(self.params['input_dim'] * math.sqrt(2) / min_scale))
        in_spacing = math.pow(2, self.params['input_level'])

        # to keep resample / augmentation easy. Truth chip will have same slide size as chip.
        # (even though convolution/rf_size shrinks the output)
        # TODO: Check if roundoff is a problem with truth alignment.
        out_chip_dim = int(in_chip_dim / self.params['rf_stride'])
        out_spacing = in_spacing * self.params['rf_stride']
        
        # Crop the error map to make sure that truth chips are always in the truth map.
        margin = int(math.ceil(out_spacing * out_chip_dim / 2))
        #error_map = self.crop_error_map(error_map, truth_map, margin)

        region = truth_map.region
        region = [region[0] + margin, region[1] + margin,
                  region[2] - margin, region[3] - margin]
        error_map.zero(region)
        
        # Get a list of sample points in the error map coordinate system.
        if True:
            # Get a list of sample centers (output / truth coordinates).
            # pylaw sample requires a normalized pdf.
            # make a list of random floats as samples.
            samples = np.random.uniform(0, 1, sample_count)
            samples.sort()
            points = np.zeros((sample_count,2))
            error_map.image = error_map.image / error_map.image.sum()
            pylaw.sample(error_map.image, samples, points)
        else:
            points, scores = pylaw2.sample(error_map.image, 50, sample_count)

        if debug_tag:
            img = error_map.image*255.0/np.max(error_map.image)
            img = img.astype(np.uint8)
            img = np.stack((img,img,img), axis=2)
            for pt in points:
                x = int(pt[1])
                y = int(pt[0])
                cv2.line(img,(x-10,y),(x+10,y),(255,255,0),5)
                cv2.line(img,(x,y-10),(x,y+10),(255,255,0),5)
            cv2.imwrite("debug/pdf_%s.png"%debug_tag, img)

        if prediction_map:
            p_spacing_x, p_spacing_y = prediction_map.get_spacing()
            p_origin = prediction_map.get_origin()

        e_spacing_x, e_spacing_y = error_map.get_spacing()
        e_origin = error_map.get_origin()
        t_spacing_x, t_spacing_y = truth_map.get_spacing()
        t_origin = truth_map.get_origin()

        new_chips = []
        for idx in range(len(points)):
            # Convert point from error coordinates to slide coordinates.
            x = e_origin[0] + (points[idx][1] * e_spacing_x) 
            y = e_origin[1] + (points[idx][0] * e_spacing_y) 
            # Get the input image chip from girder.
            image = g.get_image_cutout(gc, self.item_id, (x,y), in_chip_dim, in_chip_dim,
                                       scale=1.0/in_spacing, cache='cache')

            # if there is a prediction image, we want to add it as the fourth input
            if not(prediction_map is None):
                #crop out the section that we need for the chip
                # (x,y) is in level0 coordinates.
                px = (x - p_origin[0]) / p_spacing_x
                py = (y - p_origin[1]) / p_spacing_y
                in_x0 = int(px - (in_chip_size / 2))
                in_y0 = int(py - (in_chip_size / 2))
                prediction_chip = prediction_map.image[in_y0:in_y0+in_chip_size,
                                                       in_x0:in_x0+in_chip_size]
                #add it as the fourth channel
                image = np.dstack((image, prediction_chip))
            else:
                #if there isn't a prediction image, just add zeros as the fourth column
                image = np.dstack((image, np.ones(image.shape[:-1])))

            # Crop the corresponding section from the truth.
            # (x,y) is in level0 coordinates.
            tx = (x - t_origin[0]) / t_spacing_x
            ty = (y - t_origin[1]) / t_spacing_y
            t_x0 = int(tx - (out_chip_dim / 2))
            t_y0 = int(ty - (out_chip_dim / 2))
                            
            truth = truth_map.image[t_y0:t_y0+out_chip_dim, t_x0:t_x0+out_chip_dim,:]
            
            chip_data = ChipData(image, truth, (x,y), self, in_spacing)
            new_chips.append(chip_data)
            
        return new_chips


    # TODO: Generalize this to all targets (pos, neg, wing_class ...)
    def compute_negative_error_image(self, image, net):
        """ 
        I am debating whether to make an image ivar or pass it in as an argument.
        Using arguments makes the api more complex, but forces me to keep memory usage small.
        PDF is not normalized, and spacing is equal to output spacing.
        """
        net_out = net_utils.execute_large_image(net, image, self.params)
        
        annotations = self.annotation['aircraft']
        if annotations is None:
            annotations = []
        # Get the negative channel.
        # Invert it to compute error from activation.
        error = 1.0 - net_out[..., 0]

        # Threshold: Sort of like margin/hinge
        error = error - 0.4
        # I am not sure I like chopping off the peaks.
        # we sample the locations with max error.
        error = np.clip(error, 0.0, 0.7)

        # set all the positive regions to 0 (should we do a gaussian window?)
        # This error map is computed at "base resolution" (some chips are subsampled)
        truth_spacing = self.params['rf_stride']
        for e in annotations:
            # Radius is output coordinates.  twice the size of the plane.
            exclusion_radius = 2 * e['radius'] / truth_spacing
            # convert center from input to output coordinate system.
            x = (e['center'][0] / truth_spacing)
            y = (e['center'][1] / truth_spacing)
            cv2.circle(error, (int(x),int(y)), int(exclusion_radius), (0), thickness = -1)

        return error

            
# I think this manages all data for a training run.
class TrainingData:
    # Images are incrementally loaded and sampled.
    # This is the index of the next image to load.
    image_data_index = 0
    
    image_data = []
    sample_info = {}
    folder_path = ""
    target = ""
    #target_name_to_id = {}
    exclusion_radius = 10
    inclusion_radius = 5
    # For normalizing pdfs across multiple images.
    running_pixel_total = 1.0
    running_pixel_count = 1.0
    decay_factor = 0.98
    
    # Read the dictionary to convert annotation names to their numeral ids.
    #def read_target_ids(self, params):
    #    self.target_name_to_id ={}
    #    with open(os.path.join(params['folder_path'], 'Annotation-Labels.csv')) as csvfile:
    #        readCSV = csv.reader(csvfile, delimiter=',')
    #        for row in readCSV:
    #            self.target_name_to_id[row[1].lower()] = int(row[0])
                
    def __init__(self, params):
        """
        params['level']: 0=>full res,  1=>half, 2=>quarter
        """
        if not 'input_level' in params:
            params['input_level'] = 0
        if not 'input_spacing' in params:
            params['input_spacing'] = math.pow(2.0,params['input_level'])
        
        # Save the highest scoring negative chip from each image.
        # Potential false negative.
        self.debug_chips = []

        # Why not just store all the chips here.
        self.pos_chips = []
        self.neg_chips = []

        self.folder_path = params['folder_path']
        self.params = params
        # Annotation point to error map.
        self.exclusion_radius = 10
        self.inclusion_radius = 5

        # The images are load on demand.  Make the empty structure to hold info for each image
        self.image_data = []

        # Positives are easy.  Just load anntoations.
        # Negatives:
        # - explicit negatives (marked by proofreader)
        # - False positives found by processing fully annotated images.
        
        # two types of images:
        # - complete
        # - in progress

        print("working")
        # Hard code the single "free tail bat" slide (HEC-1606 Slide F
        image_data = ImageData("5915da6add98b578723a09cb", params)
        self.image_data.append(image_data)
        
        
        """
        for g_folder_id in params['working_folder_ids']:
            image_ids = self.get_folder_image_ids(g_folder_id)
            # load all the working folders up front.
            for item_id in image_ids:
                # should I save these items to revist later too?
                # I woud need to mark them asn incomplete and not mine for false positives.
                image_data = ImageData(item_id, params)
                self.pos_chips += image_data.get_annotation_chips('aircraft')
                self.neg_chips += image_data.get_annotation_chips('negative')
                print("%s,  %d,  %d"%(item_id, len(self.pos_chips), len(self.neg_chips)))
                
        # Collect all the ids for images loaded on demand,
        # completely annotated images that can be used for hard negative mining.
        for g_folder_id in params['finished_folder_ids']:
            image_ids = self.get_folder_image_ids(g_folder_id)
            for item_id in image_ids:
                image_data = ImageData(item_id, params=params)
                self.image_data.append(image_data)
        random.shuffle(self.image_data)
        """
                
        
    # Load one more image.
    def incremental_load(self):
        gc = g.get_gc()
        image_data = self.image_data[self.image_data_index]
        input_spacing = math.pow(2, self.params['input_level'])
        output_spacing = self.params['rf_stride'] * input_spacing
        
        # Get and scale the prediction.
        prediction_map = g.get_heatmap_image(image_data.item_id,
                                             self.params['prediction_heatmap'])        
        if prediction_map:
            # convert this to intput pixel size. (Will this be a problem?)
            prediction_map.set_spacing((input_spacing, input_spacing))

        # Get and scale the truth map.
        truth_map = g.get_heatmap_image(image_data.item_id, self.params['truth_heatmap'])        
        # convert this to output pixel size.
        truth_map.set_spacing((output_spacing, output_spacing))

        # Now for the error/sample map.
        np_error_map =  g.get_heatmap_image(image_data.item_id, self.params['error_heatmap'])
        if np_error_map is None:
            # Split up the error map into positive and negative single channels.
            # The error maps and truth maps are rgba channels (_, pos, neg, dont_care)
            # This is a little ugly.
            neg_emap = truth_map.get_channel_map(0)
            neg_emap.invert()
            pos_emap = truth_map.get_channel_map(1)
        else:
            pos_emap = np_error_map.get_channel_map(1)
            neg_emap = np_error_map.get_channel_map(2)
            
        debug_tag = None
        if 'debug' in self.params and 'pdf' in self.params['debug']:
            debug_tag = 'neg'
        self.neg_chips += image_data.sample_chips(self.params['chips_per_epoch'],
                                                  neg_emap, truth_map, prediction_map,
                                                  debug_tag=debug_tag)
        debug_tag = None
        if 'debug' in self.params and 'pdf' in self.params['debug']:
            debug_tag = 'pos'
        self.pos_chips += image_data.sample_chips(self.params['chips_per_epoch'],
                                                  pos_emap, truth_map, prediction_map,
                                                  debug_tag=debug_tag)

        # Move to the next image to load.
        self.image_data_index += 1
        if self.image_data_index >= self.get_number_of_images():
            self.image_data_index = 0


    def save_chips(self):
        """
        Starting training over causes catestrophic forgetting.
        This saves the current set of trainine chips (and errors)
        so the next time we train, we will start with the same
        training data. (Like restart).
        Use a directory of pngs and save errors in the filename.
        """
        self.save_chips_in_girder()
        self.chip_dirname = "chips"
        if os.path.isdir(self.chip_dirname):
            # remove all the previous chips.
            if os.path.isdir('tmp'):
                shutil.rmtree('tmp')
            shutil.move(self.chip_dirname, 'tmp')

        ensuredir(self.chip_dirname)
        ensuredir(os.path.join(self.chip_dirname, 'pos'))
        ensuredir(os.path.join(self.chip_dirname, 'neg'))

        self.pos_chips = sorted(self.pos_chips, key=lambda chip: chip.error, reverse=True)
        self.neg_chips = sorted(self.neg_chips, key=lambda chip: chip.error, reverse=True)
        
        pos_chips = self.pos_chips[0:1000]
        neg_chips = self.neg_chips[0:1000]

        print("SaveChips: pos %d, neg %d =============="%(len(pos_chips), len(neg_chips)))

        for chip in pos_chips:
            chip.save(self.chip_dirname, 'pos')
        for chip in neg_chips:
            chip.save(self.chip_dirname, 'neg')
        


    def save_chips_in_girder(self):
        """
        Save the chips in a light box, so I can fix annotation.
        TODO: Clear out prvious chips. EMpty directory.
        """
        # TODO: get rid of this hard coded folder id.
        folder_id = '5cf3345070aaa9038e9b9450'

        pos_chips = [c for c in self.pos_chips if c.error > 0.1]
        neg_chips = [c for c in self.neg_chips if c.error > 0.1]

        print("SaveChips in girder: pos %d, neg %d =============="%(len(pos_chips), len(neg_chips)))

        # Create positive and negative lightbox items.
        gc = g.get_gc()
        item = gc.createItem(folder_id, 'pos', "", reuseExisting=True)

        meta_chips = []
        for chip in pos_chips:
            if chip.image_data is None:
                continue
            if chip.image_data.item_id is None:
                continue
            rad = 116 * chip.spacing / 2
            cx = chip.center[0]
            cy = chip.center[1]
            meta_chip = {'imageId':chip.image_data.item_id,
                         'region':[cx-rad,cy-rad, cx+rad,cy+rad]}
            meta_chips.append(meta_chip)
        # Build up the chip meta data.
        meta = {'SimpleLightBox': meta_chips}
        gc.addMetadataToItem(item['_id'], meta)

        item = gc.createItem(folder_id, 'neg', "", reuseExisting=True)
        meta_chips = []
        for chip in neg_chips:
            if chip.image_data is None:
                continue
            if chip.image_data.item_id is None:
                continue
            rad = 116 * chip.spacing / 2
            cx = chip.center[0]
            cy = chip.center[1]
            meta_chip = {'imageId':chip.image_data.item_id,
                         'region':[cx-rad,cy-rad, cx+rad,cy+rad]}
            meta_chips.append(meta_chip)
        # Build up the chip meta data.
        meta = {'SimpleLightBox': meta_chips}
        gc.addMetadataToItem(item['_id'], meta)

        item = gc.createItem(folder_id, 'top', "", reuseExisting=True)
        meta_chips = []
        for chip in self.debug_chips:
            if chip.image_data is None:
                continue
            if chip.image_data.item_id is None:
                continue
            rad = 116 * chip.spacing / 2
            cx,cy = chip.center
            meta_chip = {'imageId':chip.image_data.item_id,
                         'region':[cx-rad,cy-rad, cx+rad,cy+rad]}
            meta_chips.append(meta_chip)
        # Build up the chip meta data.
        meta = {'SimpleLightBox': meta_chips}
        gc.addMetadataToItem(item['_id'], meta)
        #self.debug_chips = []


    def load_chips(self, params):
        """
        Starting training over causes catestrophic forgetting.
        This saves the current set of trainine chips (and errors)
        so the next time we train, we will start with the same
        training data. (Like restart).
        Use a directory of pngs and save errors in the filename.
        """
        self.chip_dirname = "chips"
        if not os.path.isdir(self.chip_dirname):
            return

        pos_chip_filenames = glob.glob("%s/pos/*.json"%self.chip_dirname)
        for filepath in pos_chip_filenames:
            chip = ChipData()
            if chip.load(filepath, self, params):
                self.pos_chips.append(chip)
                #if not chip.is_positive():
                #    print("Misclassified as positive: %s"%filepath)
        
        print('loading negative chips')
        neg_chip_filenames = glob.glob("%s/neg/*.json"%self.chip_dirname)
        for filepath in neg_chip_filenames:
            chip = ChipData()
            if chip.load(filepath, self, params):
                self.neg_chips.append(chip)
                #if chip.is_positive():
                #    print("Misclassified as negative: %s"%filepath)

        print("Loaded chips: pos %d, neg %d"%(len(self.pos_chips), len(self.neg_chips)))
                
    


        
    def prune_chips(self):
        # Print the overall performance
        total = len(self.pos_chips) + len(self.neg_chips)
        tmp = [c.error for c in self.pos_chips]
        tmp = np.array(tmp) > 0.1
        errors = np.sum(tmp)
        tmp = [c.error for c in self.neg_chips]
        tmp = np.array(tmp) > 0.1
        errors += np.sum(tmp)
        print("Percentage correct: %d"%(int(100*(total-errors)/total)))
        
        if len(self.neg_chips) > 0 :
            chips = sorted(self.neg_chips, key=lambda chip: chip.error, reverse=True)
            self.neg_chips = chips[0:self.params['max_num_training_images']]
            print("%d neg chips, error range: %f to %f"%(len(chips), chips[-1].error, chips[0].error))
            if 'debug' in self.params:
                debug_dir = os.path.join(".", self.params['target_group'], 'debug')
                # save out the top 10 offending chips.
                for idx in range(min(10,len(self.neg_chips))):
                    self.neg_chips[idx].save_debug_images(debug_dir)
                
        if len(self.pos_chips) > 0 :
            num = max(self.params['max_num_training_images'], int(0.9*len(self.pos_chips)))
            chips = sorted(self.pos_chips, key=lambda chip: chip.error, reverse=True)
            self.pos_chips = chips[0:num]
            print("%d pos chips, error range: %f to %f"%(len(self.pos_chips), chips[-1].error, chips[0].error))
            if 'deubg' in self.params:
                debug_dir = os.path.join(".", self.params['target_group'], 'debug')
                # save out the top 10 offending chips.
                for idx in range(min(10,len(self.pos_chips))):
                    self.pos_chips[idx].save_debug_images(debug_dir)

        # Print the overall performance
        total = len(self.pos_chips) + len(self.neg_chips)
        tmp = [c.error for c in self.pos_chips]
        tmp = np.array(tmp) > 0.1
        errors = np.sum(tmp)
        tmp = [c.error for c in self.neg_chips]
        tmp = np.array(tmp) > 0.1
        errors += np.sum(tmp)


    def get_number_of_images(self):
        # I am starting to support load on demand.
        return len(self.image_data)


    def get_number_of_positive_chips(self):
        return len(self.pos_chips)


    def get_number_of_negative_chips(self):
        return len(self.neg_chips)

    
    def get_image_data_from_root(self, root):
        for image_data in self.image_data:
            if image_data.root == root:
                return image_data
        return None

            
    def load_positive_chips(self, image_data, image, chip_size):
        chips = image_data.load_positive_chips(image, chip_size, self.params)
        self.pos_chips += chips



        
    def sample_negative_chips(self, image_data, image,  chip_size, net):
        """
        Compute the negative error map for this image (used for the sampling pdf)
        It is just the net response with plane circles zeroed out.
        Sample 100 chips, add to negative chip collection, and prune low error chips away.
        """

        # Net is responding to circles!  Thorugh some in to get rid of them.
        image = image.copy()
        for i in range(10):
            rx = image.shape[1] - 400
            ry = image.shape[0] - 400
            center = (int(200+random.random()*rx), int(200+random.random()*ry))
            radius = 80 + int(random.random()*40)
            cv2.circle(image, center, radius, [128,128,128], -1)


        net_out = net_utils.execute_large_image(net, image, self.params)
        
        emap = image_data.compute_negative_error_image(net_out, self.params)

        total_num_pos = max(1, self.get_number_of_positive_chips())

        # Sample some chips and add them to the image negative chip array.
        # We do not want to sample too many chips.  If we sample the same error more than once,
        # we will prune too many other chips.  We could add a proximity condition when sampling.
        chips = image_data.sample_negative_chips(image, emap, chip_size, 100, self.params)

        if len(chips) > 0:
            self.debug_chips.append(chips[0])

        self.neg_chips += chips

        max_error = 0.0
        if len(self.neg_chips) > 0 :
            chips = sorted(self.neg_chips, key=lambda chip: chip.error, reverse=True)
            self.neg_chips = chips[0:total_num_pos]
            max_error = self.neg_chips[0].error

        # save out the largest error chips for debugging.  Looking at offenders after training
        # may not work because any input can get squashed down.
        #debug_dir = os.path.join(".", self.params['target_group'], 'debug')
        #ensuredir(debug_dir)
        #chip = self.neg_chips[0]
        #chip.save(debug_dir, 'neg')
            
        # This is just to monitor to see how the negative samples accummilate.
        # 1.5 times more negative than positive?
        total_num_neg = self.get_number_of_negative_chips()
        print("Total pos chips: %d, total neg chips(%f): %d"%(total_num_pos, max_error, total_num_neg))

            
        
    def choose_chips(self, chips, num_samples):
        """
        This returns a set of chips randomly selected with probability based on their error.
        A single chip may be repeated multiple times in the reutrned list if it has a high
        relative probability.
        """
        chips.sort(key=lambda chip: chip.error)
        # This does a similar algorythm to pylaw.sample
        total = sum([c.error for c in chips])
        samples = list(np.random.uniform(0, total, num_samples))
        samples.sort()
        selected_chips = []
        running_total = 0.0
        for chip in chips:
            running_total += chip.error
            while len(samples) > 0 and samples[0] < running_total:
                del samples[0]
                selected_chips.append(chip)
        return selected_chips
        

    def sample_batch(self, params):
        """
        select pos and neg chips based on their error.
        Augment and generate batch input and truth images.
        Remeber the chips so the error can be updated.
        NOTE: I may allow more negative chips than positive, 
        so computing probability from error is not a simple normalization.
        Just 50/50 split positive to negative for now.
        """
        inputs = []
        truths = []
        ignores = []

        num = params['batch_size']
        input_dim = params['input_dim']
        # Size of the networks output image given the input size.
        # This considers stride and boundary shrinkage.
        truth_dim = int((input_dim - params['rf_size'] + params['rf_stride']) /
                        params['rf_stride'])
        
        # Positive
        # Create a single list of chips
        pos_chips = self.pos_chips
        pos_chips = self.choose_chips(pos_chips, int(num/2))
        for chip in pos_chips:
            input, truth, ignore = chip.augment(input_dim, truth_dim, params)
            # some interpolation is causeing some values to be less that 1
            inputs.append(input)
            truths.append(truth)
            ignores.append(ignore)

        # Negative
        # Create a single list of chips
        neg_chips = self.neg_chips
        neg_chips = self.choose_chips(neg_chips, int(num/2))
        for chip in neg_chips:
            input, truth, ignore = chip.augment(input_dim, truth_dim, params)
            inputs.append(input)
            truths.append(truth)
            ignores.append(ignore)

        # Save so the program can update chip errors.
        self.batch_chips = pos_chips + neg_chips
    
        if 'debug' in params and 'batch2' in params['debug']:
            for idx in range(len(inputs)):
                image = inputs[idx][:,:,0:3]
                cv2.imwrite("debug/batch_%d_image.png"%idx, image)
                prediction = inputs[idx][:,:,3]
                cv2.imwrite("debug/batch_%d_image.png"%idx, prediction)
                truth = truths[idx]
                tmp = truth * 255
                cv2.imwrite("debug/batch_%d_truth.png"%idx, tmp)
                ignore = ignores[idx]
                tmp = ignore * 255
                cv2.imwrite("debug/batch_%d_ignore.png"%idx, tmp)

        return np.array(inputs), np.array(truths), np.array(ignores)
        
        
    def record_error(self, loss_np, decay):
        """
        decay = 1 =>  just set the error to the new error (no memory)
        """
        for idx in range(len(loss_np)):
            error = np.mean(loss_np[idx])
            chip = self.batch_chips[idx]
            chip.error = (1-decay)*chip.error + decay*error


    def test_positive_chips(self):
        dirpath = os.path.join(self.params['folder_path'], 'debug')
        ensuredir(dirpath)
        for image_data in self.image_data:
            root = os.path.split(image_data.file_root)[1]
            chip_size = int(math.ceil(self.params['input_dim'] * math.sqrt(2)))
            image = image_data.load_image()
            if image is None:
                return
            chips = image_data.load_positive_chips(image, chip_size, self.params)

            for chip in chips:
                x = chip.center[0]
                y = chip.center[1]
                file_path = os.path.join(dirpath, "%s_x%d_y%d_chip.png"%(root, x, y))
                cv2.imwrite(file_path, chip.chip)
                # TODO: Save out the network response to this chip.

    #------------------------------------------------------------------------------------
    # Code to load positive chips directly from Girder.

    def get_folder_image_ids(self, folder_id):
        image_ids = []
        gc = g.get_gc()
        # look for all sub folders
        resp = gc.get("folder?parentId=%s&parentType=folder"%folder_id)
        for f in resp:
            image_ids += self.get_folder_image_ids(f['_id'])
        
        # look for all items
        resp = gc.get("item?folderId=%s"%folder_id)
        for item in resp:
            if 'largeImage' in item:
                image_ids.append(item['_id'])
        return image_ids
                

    def get_collection_image_id(self, collection_id):
        image_ids = []
        # look for all sub folders
        gc = g.get_gc()
        resp = gc.get("folder?parentId=%s&parentType=collection"%collection_id)
        for f in resp:
            image_ids += self.get_folder_image_ids(f['_id'])
        return image_ids

            
        
#------------------------------------------------------------------------------------
    
                
        


    
if __name__ == '__main__':
    params = {}
    params['folder_path'] = '.'
    params['chip_size'] = 128
    params['rf_stride'] = 4
    params['rf_size'] = 92
    params['truth_radius'] = 16
    data = TrainingData(params)
    # api has changed.  make a new test.
    for image_data in data.image_data:
        if image_data.file_root == './train/Moscow/3857_9_309_159_20170202_31e1ec5c-62ba-4fbe-9a1f-b7c0a1dc908b':
            break

    image = image_data.load_image()        
    chip_size = 312
    data.load_positive_chips(image_data, image, chip_size)
