from pprint import pprint
import numpy as np
import cv2
import scipy
import scipy.misc
import cv2
import os
import sys
import math
import girder_client
import ipdb
if sys.version_info[0] < 3:
    import urllib2
else:
    import urllib.request
    import urllib.error
    
import my_utils

GIRDER_URL = 'https://images.slide-atlas.org'
GIRDER_USERNAME = 'law12019'
GIRDER_KEY = 'MJyaKIJxIkAb8l7OS7mffPE7QvB8H1WxDpyllzcG'


GIRDER_CLIENT = None



class Heatmap:
    """
    Overlay image that has its own region / spacing.
    """
    
    # I am not sure about storing the image data here.  The image data references the chip.
    def __init__(self, image=None, region=None):
        self.image = image
        self.region = region
        # minx,miny, maxx,maxy
        
    def load_from_girder(self, annot_id, cache='cache', gc=None):
        gc = get_gc(gc)

        resp = gc.get("annotation/%s"%annot_id)
        annot = resp['annotation']
        if len(annot['elements']) == 0:
            return False
        element = annot['elements'][0]
        if element['type'] != "rectangle" or not 'user' in element:
            return None
        user = element['user']
        file_id = None
        if 'imageFileId' in user:
            file_id = user['imageFileId']
        elif 'imageUrl' in user:
            file_id = user['imageUrl'].split('/')[6]
        if file_id is None:
            return None

        img = get_file_image(gc, file_id=file_id, cache='cache')
        center = element['center']
        width = element['width']
        height = element['height']
        left = int(center[0]-(width/2))
        right = int(left + width)
        top = int(center[1]-(height/2)) 
        bottom = int(top + height)

        self.image = img
        self.region = (left, top, right, bottom);
        self.annotation_id = annot_id
        self.file_id = file_id
        return True

    def get_origin(self):
        return (self.region[0], self.region[1])

    def set_spacing(self, spacing):
        truth_spacing_x, truth_spacing_y = self.get_spacing()
        scale_x = truth_spacing_x / spacing[0]
        scale_y = truth_spacing_y / spacing[1]
        self.image = cv2.resize(self.image,None,fx=scale_x,fy=scale_y)

    def invert(self):
        self.image = np.invert(self.image)
        return self

    def zero(self, region, outside=True):
        # Region (full res) coordinates are converted to image.
        if self.image is None:
            return
        spacing_x, spacing_y = self.get_spacing()
        if not outside:
            # Inside is a single rectangle.
            startx = max(0,int(region[0]/spacing_x))
            starty = max(0,int(region[1]/spacing_y))
            endx = min(self.image.shape[1], int(region[2]/spacing_x))
            endy = min(self.image.shape[0], int(region[3]/spacing_x))
            self.image[starty:endy,startx:endx,...] = 0
            return

        # Outside is a bit more complex with 4 rectangles.
        start = self.region[0]
        end = region[0]
        if start < end:
            end = min(end, self.region[2])
            start = int(start / spacing_x)
            end = int(end / spacing_x)
            self.image[:,start:end,...] = 0
        start = region[2]
        end = self.region[2]
        if start < end:
            start = max(start, self.region[0])
            start = int(start / spacing_x)
            end = int(end / spacing_x)
            self.image[:,start:end,...] = 0

        start = self.region[1]
        end = region[1]
        if start < end:
            end = min(end, self.region[3])
            start = int(start / spacing_y)
            end = int(end / spacing_y)            
            self.image[start:end,...] = 0
        start = region[3]
        end = self.region[3]
        if start < end:
            start = max(start, self.region[1])
            start = int(start / spacing_y)
            end = int(end / spacing_y)            
            self.image[start:end,...] = 0

    
    def get_channel_map(self, channel):
        if len(self.image.shape) != 3:
            print("Cannot get channel map: bad image")
            return None
        if channel < 0 or channel >= self.image.shape[2]:
            print("Cannot get channel map. bad channel")
            return None
        channel_map = Heatmap()
        channel_map.image = self.image[:,:,channel].copy()
        # Region is a tuple so no need to copy
        channel_map.region = self.region
        return channel_map
        
    def get_spacing(self):
        if self.image is None:
            return None
        spacing_x = (self.region[2] - self.region[0]) / self.image.shape[1]
        spacing_y = (self.region[3] - self.region[1]) / self.image.shape[0]
        return (spacing_x, spacing_y)

    def save_to_girder(self, item_id, name, gc=None):
        gc = get_gc(gc)

        if self.image is None:
            return False

        if self.region is None:
            resp = gc.get('item/%s/tiles'%item_id)
            self.region = (0, 0, resp['sizeX'], resp['sizeY'])
        
        # Upload the image as a file to the item.
        # No time to debug the direct upload ....
        success, data_np = cv2.imencode(".png", self.image)
        if not success:
            return False
        data = data_np.tobytes()

        # If the file exists, replace it.
        file_obj = get_file(item_id, name, gc=gc)
        if file_obj is None:
            params = {
                'parentType': 'item',
                'parentId': item_id,
                'name': name,
                'size': data_np.shape[0],
                'mimeType':"image/png"
            }
            resp = gc.post('file', params)
        else:
            params = {
                'size': data_np.shape[0],
            }
            resp = gc.put('file/%s/contents' % file_obj['_id'], params)

        obj = gc.post(
            'file/chunk?offset=%d&uploadId=%s' % (0, resp['_id']),
            data=data)        

        # now to create the annotation.
        cx = (self.region[0] + self.region[2]) * 0.5
        cy = (self.region[1] + self.region[3]) * 0.5
        w = self.region[2] - self.region[0]
        h = self.region[3] - self.region[1]
        url = GIRDER_URL + '/api/v1/file/%s/download?contentDisposition=inline'%obj['_id']
        e = {"rotation":0,
             "center":[cx,cy,0],
             "width": w,
             "height": h,
             "lineWidth":0,
             "lineColor":"#00ffff",
             "type":"rectangle",
             "user":{"imageUrl": url,
                     "imageFileId": obj['_id']}}
        annot = {"name": name,
                 "elements": [e]} 

        annot_id = get_annotation_id_from_name(item_id, name, gc)
        if annot_id is None:
            resp = gc.post('annotation?itemId=%s'%item_id, json=annot)
        else:
            resp = gc.put('annotation/%s'%annot_id, json=annot)
            


        


#===============================================================================
# heat map stuff



def get_heatmap_image(item_id, name, cache='cache', gc=None):
    """
    return an object containing the image, and metadata
    """
    gc = get_gc(gc)
    resp = gc.get("annotation?itemId=%s&name=%s"%(item_id,name))
    if len(resp) == 0:
        return None
    annot_id = resp[0]['_id']

    heatmap = Heatmap()
    if heatmap.load_from_girder(annot_id, gc=gc):
        return heatmap

    return None


def get_file(item_id, file_name, gc=None):
    gc = get_gc(gc)

    resp = gc.get("item/%s/files"%item_id)
    for file_obj in resp:
        if file_obj['name'] == file_name:
            return file_obj
    return None
            
    
# Should we create an alpha channel if one does not exists?
def file_to_heatmap(item_id, file_name, region=None, gc=None):
    """
    Turn a file, already uploaded in an item, into a heatmap.
    region is (xmin, ymin, xmax, ymax)
    """
    gc = get_gc(gc)

    if region is None:
        # Put the heatmap over the whole image.
        resp = gc.get("item/%s/tiles"%item_id)
        region = (0, 0, resp['sizeX'], resp['sizeY'])

    # Get the file id.
    file_obj = get_file(item_id, file_name, gc)
    if file_obj is None:
        return
    file_id = file_obj['_id']

    # create the heatmap annotation
    center = [(region[0]+region[2])*0.5, (region[1]+region[3])*0.5, 0]
    width = region[2] - region[0]
    height = region[3] - region[1]

    url = GIRDER_URL + '/api/v1/file/%s/download?contentDisposition=inline'%file_id
    annot = {"elements": [{"center": center,
                           "height": height,
                           "width": width,
                           "user": {'imageUrl': url,
                                    'imageFileId': file_id},
                           "lineColor":"#00ffff",
                           "rotation":0,
                           "lineWidth":0,
                           "type":"rectangle"}],
             "name": file_name}
    gc.post("annotation?itemId=%s"%item_id, json=annot)
    

# legacy
def get_image_file(gc, item_id, filename, cache='cache'):
    return get_file_image(gc, item_id=item_id, filename=filename, cache=cache)


    
def get_file_image(gc, file_id=None, item_id=None, filename=None, cache='cache'):

    """
    Either pass in the file_id, or the item_id and filename.
    This is for loading an arbitrary image file from a girder item.
    Just give the item id and the name of the image file you want.
    A numpy array is returned.
    """

    if file_id is None:
        if filename is None:
            return None
        file_obj = get_file(item_id, filename, gc=None)
        if file_obj is None:
            return None
        file_id = file_obj['_id']


    url = GIRDER_URL + "/api/v1/file/" + file_id + "/download" #?contentDisposition=attachment
    if sys.version_info[0] < 3:
        req = urllib2.Request(url)
        req.add_header('Girder-Token', gc.token)
        try:
            resp = urllib2.urlopen(req)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
            return image
        except urllib2.HTTPError as err:
            if err.code == 400:
                print("Bad request!")
            elif err.code == 404:
                print("Page not found!")
            elif err.code == 403:
                print("Access denied!")
            else:
                print("Something happened! Error code %d" % err.code)
    else:
        req = urllib.request.Request(url)
        req.add_header('Girder-Token', gc.token)
        try:
            resp = urllib.request.urlopen(req)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
            return image
        except urllib.error.HTTPError as err:
            if err.code == 400:
                print("Bad request!")
            elif err.code == 404:
                print("Page not found!")
            elif err.code == 403:
                print("Access denied!")
            else:
                print("Something happened! Error code %d" % err.code)
    return None




#===============================================================================






class GirderCache:
    """
    Avoid having to read annotation more than once when loading multiple classes / labels.
    """
    def __init__(self, gc=None):
        self.GC = get_gc(gc)
        self.token = self.GC.token
        self.Cache = {}

    def get(self, req):
        if req in self.Cache:
            return self.Cache[req] 
        resp =  self.GC.get(req)
        self.Cache[req] = resp
        return resp
    
    def put(self, req):
        return self.GC.put(req)

    def post(self, req):
        return self.GC.put(req)

    def delete(self, req):
        return self.GC.put(req)


#===============================================================================


def get_gc(gc=None, server="lemon"):
    global GIRDER_CLIENT
    global GIRDER_URL
    global GIRDER_KEY
    if server == "images":
        GIRDER_URL = 'https://images.slide-atlas.org'
        GIRDER_KEY = 'MJyaKIJxIkAb8l7OS7mffPE7QvB8H1WxDpyllzcG'
        
    if gc is None:
        if GIRDER_CLIENT is None:
            GIRDER_CLIENT = girder_client.GirderClient(apiUrl= GIRDER_URL+'/api/v1')
            GIRDER_CLIENT.authenticate(GIRDER_USERNAME, apiKey=GIRDER_KEY)
        gc = GIRDER_CLIENT
    return gc









# Upload the image to girder (for debugging)
def upload_image(image, item_name, destination_folder_id,
                 gc=None, stomp=True):
    gc = get_gc(gc)
    resp = gc.get("item?folderId=%s&name=%s&limit=50&offset=0&sort=lowerName&sortdir=1" \
                  %(destination_folder_id, item_name))
    if not stomp or len(resp) == 0:
        girder_item = gc.createItem(destination_folder_id, item_name,
                                    "debugging image")
    else:
        girder_item = resp[0]
        resp = gc.get("item/%s/files?limit=500"% girder_item['_id'])
        for f in resp:
            gc.delete("file/%s"%f['_id'])
    gc.addMetadataToItem(girder_item['_id'], {'lightbox': 1})

    output_path = '/tmp'
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tmp_file_name = os.path.join(output_path, 'image.png')
    #scipy.misc.imsave(tmp_file_name, image)
    cv2.imwrite(tmp_file_name, image)
    # upload the file into the girder item
    gc.uploadFileToItem(girder_item['_id'], tmp_file_name)


# Batch is numpy (batch, comps, dimy, dimx)
# Upload the images to girder (for debugging)
# Crude, but at least the chips do not change when the annotation changes.
def upload_images(images, item_name, destination_folder_id, \
                  num=100, gc=None, stomp=True, filenames=None, \
                  description=''):
    gc = get_gc(gc)
    resp = gc.get("item?folderId=%s&name=%s&limit=50&offset=0&sort=lowerName&sortdir=1" \
                  %(destination_folder_id, item_name))
    if not stomp or len(resp) == 0:
        girder_item = gc.createItem(destination_folder_id, item_name, \
                                    description=description)
    else:
        girder_item = resp[0]
        gc.put("item/%s"%girder_item['_id'], parameters={'description':description})
        resp = gc.get("item/%s/files?limit=500"% girder_item['_id'])
        for f in resp:
            gc.delete("file/%s"%f['_id'])
    gc.addMetadataToItem(girder_item['_id'], {'lightbox': 1})

    total = len(images)
    if total == 0:
        return
    tmp = total
    if num != None and num < total:
        tmp = num

    for i in range(tmp):
        image = images[i]
        output_path = '/tmp'
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if filenames == None:
            tmp_file_name = os.path.join(output_path, '%03d.png'%i )
        else:
            tmp_file_name = os.path.join(output_path, filenames[i] )
        print(tmp_file_name, image.shape)
        #scipy.misc.imsave(tmp_file_name, image)
        cv2.imwrite(tmp_file_name, image)
        # upload the file into the girder item
        gc.uploadFileToItem(girder_item['_id'], tmp_file_name)

        
        

def get_collection_item_id_from_name(collection_name, item_name, gc=None):
    gc = get_gc(gc)
    resp = gc.get('collection?text=%s&limit=1&offset=0'%collection_name)
    assert len(resp) > 0, "%d collection not found"%collection_name
    collection_id = resp[0]['_id']
    return get_decendant_item_id(collection_id, 'collection', item_name, gc)


# give names of the collection and folder and return the first id found
def get_collection_folder_id_from_name(collection_name, folder_name, gc=None):
    gc = get_gc(gc)
    resp = gc.get('collection?text=%s&limit=1&offset=0'%collection_name)
    assert len(resp) > 0, "%s collection not found"%collection_name
    collection_id = resp[0]['_id']
    return get_decendant_folder_id(collection_id, 'collection', folder_name, gc)


# given a folder name and parent folder id return:
# [image_ids], folder_id
def get_image_ids_from_folder_name(folder_name, parent_folder_id, gc=None):
    gc = get_gc(gc)
    # get the folder id from its name.
    folder_id = get_decendant_folder_id(digital_globe_images_folder_id,
                                        'folder', folder_name, gc)
    assert folder_id, ("Could not find folder %s"%folder_name)
    return get_image_ids_from_folder_id(folder_id, gc), folder_id


def get_image_ids_from_folder_id(folder_id, gc=None):
    gc = get_gc(gc)
    # get a list of images in the folder
    image_ids = []
    resp = gc.get("item?folderId=%s&limit=1000&offset=0&sort=lowerName&sortdir=1"%folder_id)
    for item in resp:
        # todo: fix this bug.  Small images do not have this so are skipped.
        if "largeImage" in item:
            image_ids.append(item['_id'])
    return image_ids


# create a named annotation or return its id.
def create_or_find_annotation(image_id, name, gc=None):
    gc = get_gc(gc)
    annot_id = get_annotation_id_from_name(image_id, name, gc)
    if annot_id:
        return annot_id
    else:
        annot = {"elements":[],"name":name}
        resp = gc.post("annotation", parameters={"itemId":image_id}, json=annot)
        return resp["_id"]

# Get an annotation id from its name.
# returns None if doesnot exist for item
def get_annotation_id_from_name(image_id, name, gc):
    gc = get_gc(gc)
    resp = gc.get("annotation?itemId=%s&name=%s" % (image_id, name))
    if len(resp) > 0:
        return resp[0]["_id"]
    else:
        return None

# get an annotation location from its id
# returns center, width, and height
def get_annotation_loc_from_id(annotation_id, gc):
    gc = get_gc(gc)
    resp = gc.get("annotation/%s" %annotation_id)
    if len(resp) > 0:
        elements = resp['annotation']['elements'][0]
        return elements['center'],elements['width'],elements['height']
    else:
        return None
    
# adds an image to a heatmap annotation using their ids
def add_image_to_annotation(annotation_id, image_id, gc):
    gc = get_gc(gc)
    resp = gc.get("annotation/%s" %annotation_id)
    if len(resp) > 0:
        elements = resp['annotation']['elements'][0]
        elements.update({'user':{'imageUrl':"https://images.slide-atlas.org/api/v1/file/%s/download?contentDisposition=inline"%image_id}})
        gc.put("annotation/%s" %annotation_id,json=resp['annotation'])


# returns the first decendant image with a matching name.
def get_decendant_item_id(ancestor_id, ancestor_type, item_name, gc=None):
    gc = get_gc(gc)
    if ancestor_type == 'folder':
        # look for the item
        resp = gc.get('item?folderId=%s&name=%s&limit=50'%(ancestor_id, item_name))
        if len(resp) > 0:
            return resp[0]['_id']

    resp = ['do while']
    offset = 0
    while len(resp) > 0:
        resp = gc.get('folder?parentType=%s&parentId=%s&limit=50&offset=%d'%(ancestor_type,ancestor_id,offset))
        offset += 50
        for folder in resp:
            item_id = get_decendant_item_id(folder['_id'], 'folder', item_name, gc)
            if item_id:
                return item_id
    return None


# returns the first decendant folder with a matching name.
def get_decendant_folder_id(ancestor_id, ancestor_type, folder_name, gc=None):
    gc = get_gc(gc)
    resp = ['do while']
    offset = 0
    while len(resp) > 0:
        resp = gc.get('folder?parentType=%s&parentId=%s&limit=50&offset=%d'%(ancestor_type,ancestor_id,offset))
        offset += 50
        for folder in resp:
            if folder['name'] == folder_name:
                return folder['_id']
            folder_id = get_decendant_folder_id(folder['_id'], 'folder', folder_name, gc)
            if folder_id:
                return folder_id
    return None


def get_image_cutout(gc, image_id, center, width, height, scale=1, cache='cache'):
    """
    Width and height are the final size of the image returned.
    center:  full res image coordinates.
    scale: > 1 implies the image is magnified (small GSD)
    """
    dx = int(round(width/2.0))
    dy = int(round(height/2.0))
    if scale == 1:
        chip_url = GIRDER_URL + "/api/v1/item/" + image_id + "/tiles/region?" + \
                   ("left=%d&top=%d&" % (center[0]-dx, center[1]-dy)) + \
                   ("regionWidth=%d&regionHeight=%d" % (width,height)) + \
                   "&units=base_pixels&encoding=JPEG&jpegQuality=95&jpegSubsampling=0"
    else:
        # does not shrink like I want.
        #chip_url = GIRDER_URL + "/api/v1/item/" + image_id + "/tiles/region?" + \
        #           ("magnification=%f&" % (40 * scale)) + \
        #           ("left=%d&top=%d&" % (left, top)) + \
        #           ("regionWidth=%d&regionHeight=%d" % (width,height)) + \
        #           "&units=base_pixels&encoding=JPEG&jpegQuality=95&jpegSubsampling=0"
        in_width = int(round(width / scale)) 
        in_height = int(round(height / scale))
        in_dx = int(round(in_width/2.0))
        in_dy = int(round(in_height/2.0))
        chip_url = GIRDER_URL + "/api/v1/item/" + image_id + "/tiles/region?" + \
                   ("left=%d&top=%d&" % (center[0]-in_dx, center[1]-in_dy)) + \
                   ("regionWidth=%d&regionHeight=%d&" % (in_width,in_height)) + \
                   ("width=%d&height=%d&" % (width,height)) + \
                   "units=base_pixels&encoding=JPEG&jpegQuality=95&jpegSubsampling=0"
        
    cache_fileroot = chip_url
    # strip off the 'http://'
    idx = cache_fileroot.find('//') + 2
    cache_fileroot = cache_fileroot[idx:]
    # replace the problemative '/'
    cache_fileroot = cache_fileroot.replace('/', '_')
    # add the cache path and extension
    cache_filepath = os.path.join(cache, "%s.png"%(cache_fileroot))    
    if os.path.isfile(cache_filepath):
        return cv2.imread(cache_filepath)

    if sys.version_info[0] < 3:
        req = urllib2.Request(chip_url)
        req.add_header('Girder-Token', gc.token)
        # intermitently does not work. Repeat and it does (not really)
        retry = 1
        while retry > 0:
            try:
                resp = urllib2.urlopen(req)
                image = np.asarray(bytearray(resp.read()), dtype="uint8")
                try:
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                except:
                    print("problem chip: %s"%chip_url)
                    return None
                # cache for fast local re read.
                cv2.imwrite(cache_filepath, image)
                return image
            except urllib2.HTTPError as err:
                if err.code == 400:
                    print("Bad request!")
                elif err.code == 404:
                    print("Page not found!")
                elif err.code == 403:
                    print("Access denied!")
                else:
                    print("Something happened! Error code %d" % err.code)
                retry -= 1
                #time.sleep(1)
                return None

    else:
        req = urllib.request.Request(chip_url)
        req.add_header('Girder-Token', gc.token)
        # intermitently does not work. Repeat and it does (not really)
        retry = 1
        while retry > 0:
            try:
                resp = urllib.request.urlopen(req)
                image = np.asarray(bytearray(resp.read()), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                # cache for fast local re read.
                cv2.imwrite(cache_filepath, image)
                return image
            except urllib.error.HTTPError as err:
                if err.code == 400:
                    print("Bad request!")
                elif err.code == 404:
                    print("Page not found!")
                elif err.code == 403:
                    print("Access denied!")
                else:
                    print("Something happened! Error code %d" % err.code)

    return None


# level: integer,  0=>highest level, 1=>half resolution ... 
def get_image(image_id, level=0, cachedir='cache'):
    my_utils.ensuredir(cachedir)
    # add the cache path and extension
    cache_filepath = os.path.join(cachedir, "%s.png"%(image_id))    
    if os.path.isfile(cache_filepath):
        return cv2.imread(cache_filepath)

    img = get_large_cutout(image_id, level)
    cv2.imwrite(cache_filepath, img)
    return img
        
        

def get_large_cutout(image_id, level=0, region=None, progress=None, gc=None):
    """
    Get a region using the tile api.
    No second recompression step, so the images should have less artifacts.
    level: integer,  0=>highest res, 1=>half resolution ... 
    region: (in the units of the level being requested) (left, top, width, height)
    """
    gc = get_gc(gc)
    if not progress:
        progress = [0.0,100.0]
    remaining = progress[1]-progress[0]
    # get the image meta data
    meta = gc.get("item/%s/tiles" % image_id)
    # girder large image,  level 0 is at the top of the pyramid.
    g_level = meta['levels'] - level - 1
    t_x = int(meta['tileWidth'])
    t_y = int(meta['tileHeight'])

    if region:
        left = int(region[0])
        top = int(region[1])
        width = int(region[2])
        height = int(region[3])
    else:
        left = 0
        top = 0
        width = int(meta['sizeX']/math.pow(2,level))-1
        height = int(meta['sizeY']/math.pow(2,level))-1
    
    # get the range of tiles needed.
    i_bds = [int(math.floor(float(left)/t_x)), \
             int(math.ceil((float(left)+width)/t_x)), \
             int(math.floor(float(top)/t_y)), \
             int(math.ceil((float(top)+height)/t_y))]

    total = float(i_bds[1]-i_bds[0])*(i_bds[3]-i_bds[2])
    count1 = 0;
    count2 = 0;
    # TODO: deal with partial tiles?
    region = np.zeros(((i_bds[3]-i_bds[2])*t_y, (i_bds[1]-i_bds[0])*t_x, 3), dtype=np.uint8)
    # Get all of the tiles and fill the region.
    print("")
    for x in range(i_bds[0], i_bds[1]):
        xo = x - i_bds[0]
        for y in range(i_bds[2], i_bds[3]):
            yo = y - i_bds[2]
            #.... get GIRDER_URL from gc ....
            tile_url = GIRDER_URL+"/api/v1/item/%s/tiles/zxy/%d/%d/%d"%(image_id,g_level,x,y)

            if sys.version_info[0] < 3:
                req = urllib2.Request(tile_url)
                req.add_header('Girder-Token', gc.token)
                count1 = count1 + 1
                count2 = count2 + 1
                try:
                    resp = urllib2.urlopen(req)
                    image = np.asarray(bytearray(resp.read()), dtype="uint8")
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    # copy into region.
                    region_img[yo*t_y:(yo+1)*t_y, xo*t_x:(xo+1)*t_x] = image
                except urllib2.HTTPError as err:
                    if err.code == 400:
                        print("Bad request!")
                    elif err.code == 404:
                        print("Page not found!")
                    elif err.code == 403:
                        print("Access denied!")
                    else:
                        print("Something happened! Error code %d" % err.code)
                    break
            else:
                req = urllib.request.Request(tile_url)
                req.add_header('Girder-Token', gc.token)
                count1 = count1 + 1
                count2 = count2 + 1
                try:
                    resp = urllib.request.urlopen(req)
                    image = np.asarray(bytearray(resp.read()), dtype="uint8")
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    # copy into region.
                    region[yo*t_y:(yo+1)*t_y, xo*t_x:(xo+1)*t_x] = image
                except urllib.error.HTTPError as err:
                    if err.code == 400:
                        print("Bad request!")
                    elif err.code == 404:
                        print("Page not found!")
                    elif err.code == 403:
                        print("Access denied!")
                    else:
                        print("Something happened! Error code %d" % err.code)
                    break

        if count2 > 100:
            count2 = 0
            print("\033[F %0.1f finished" % (progress[0] + remaining*(count1 / total)))
            
    # crop to the requested size.
    offset_y = top-(i_bds[2]*t_y)
    offset_x = left-(i_bds[0]*t_x)
    return region[offset_y:offset_y+height, offset_x:offset_x+width]


# changed this to returns sets of arrows (each set is from one image).
# I need that because a receptife field may see multiple planes at one time.
def load_arrows(folder_ids, min_scale,  gc):
    arrow_image_sets = []
    for folder_id in folder_ids:
        print('loading folder %s'%folder_id)
        # read in the arrow images.
        # Ignore dimensions.
        # make an array of  arrow / image_id pairs.
        image_ids = get_image_ids_from_folder_id(folder_id, gc);
        for image_id in image_ids:
            annot_id = get_annotation_id_from_name(image_id, "plane-nose", gc)
            if annot_id:
                resp = gc.get("annotation/%s" % annot_id)
                #image_id = resp['itemId']
                elements = resp['annotation']['elements']

                arrows = []
                for e in elements:
                    if e['type'] == 'arrow':
                        pt0 = e['points'][0]
                        pt1 = e['points'][1]
                        # fix a bug with the way arrows are saved.
                        # conversion from angle to points is wrong.
                        dy = pt1[1] - pt0[1]
                        pt1[1] = pt0[1] - dy

                        # compute and save the rotation of the plane / arrow.
                        dx = pt0[0] - pt1[0]
                        dy = pt0[1] - pt1[1]
                        e['rotation'] = math.atan2(-dy, dx) * 180 / math.pi
                        e['length'] = distance(pt0, pt1)
                        e['center'] = ((pt0[0]+pt1[0])*0.5, (pt0[1]+pt1[1])*0.5)

                        # TODO: Take length into consideration when picking a ROI
                        # crop out a region around the arrow big enough
                        # to augment without clipping corners. (max scale: double, 45 rotation).
                        # center of arrow must still be in rf.
                        dim = 543
                        dim = (int)(2.1*math.sqrt(2)* e['length'] / min_scale)

                        # Middle of the arrow.
                        mx = e['center'][0]
                        my = e['center'][1]
                        left = int(mx - dim/2)
                        top = int(my - dim/2)
                        width = int(dim)
                        height = width
                        # find the dimensions for sanity check
                        #tile_info = gc.get('/item/%s/tiles'%image_id)
                        e['item_id'] = image_id
                        # get the cropped image from girder.
                        #img = get_girder_cutout(gc, image_id, left, top, width, height)
                        img = np.zeros((100,100,3))
                        # some bad short arrows got through the annotation review process.
                        #if img.shape[0] == height and img.shape[1] == width and e['length'] > 3.0:
                        if True:
                            e['image'] = img
                            # just a large number
                            e['error'] = 100
                            arrows.append(e)
                print(len(arrows))
                arrow_image_sets.append(arrows)
    return arrow_image_sets
    

def get_image_gsd(item_id, gc=None):
    gc = get_gc(gc)
    item = gc.get('item/%s'%item_id)
    for key in item['meta']:
        if len(key) == 32:
            # It was stupid not to use a constant key for this
            # I do not even know that that code means.
            return float(item['meta'][key]["groundSampleDistance"])


# changed this to returns sets of arrows (each set is from one image).
# I need that because a receptife field may see multiple planes at one time.
def load_arrows_and_wings(folder_ids, min_scale,  gc, label_name="text-labels"):
    arrow_image_sets = load_arrows(folder_ids, min_scale, gc)

    # key: image, value: annotation elements
    annot_dict = {}

    # TODO: generalize this (almost the same code)
    
    print('looking for text labels')
    for arrow_image_set in arrow_image_sets:
        for arrow in arrow_image_set:
            item_id = arrow['item_id']
            if annot_dict.has_key(item_id):
                elements = annot_dict[item_id]
            else:
                annot_id = get_annotation_id_from_name(item_id, label_name, gc)
                if not annot_id:
                    continue
                resp = gc.get("annotation/%s" % annot_id)
                elements = resp['annotation']['elements']
                # fix a bug with the way arrows are saved.
                # conversion from angle to points is wrong.
                for e in elements:
                    if e['type'] == 'arrow':
                        pt0 = e['points'][0]
                        pt1 = e['points'][1]
                        dy = pt1[1] - pt0[1]
                        pt1[1] = pt0[1] - dy
                annot_dict[item_id] = elements

            center = arrow['center']
            radius = arrow['length']/2
            # find the wing base that is within this circle.
            for e in elements:
                if e['type'] == 'arrow':
                    # the arrow tip (anchor of the text).
                    pt0 = e['points'][0]
                    dist = distance(center, pt0)
                    if dist < radius:
                        arrow['label'] = e['label']['value']
                        break;
            if not 'label' in arrow:
                print("-----------------------------------------------------")
                print("Could not find label for arrow in image %s"%(arrow['item_id']))
                #pdb.set_trace()
                pprint(arrow['points'])
    
    annot_dict = {}
    print('looking for wings')
    for arrow_image_set in arrow_image_sets:
        count = 0
        for arrow in arrow_image_set:
            count = count + 1
            item_id = arrow['item_id']
            if annot_dict.has_key(item_id):
                elements = annot_dict[item_id]
            else:
                annot_id = get_annotation_id_from_name(item_id, "plane-wing", gc)
                if not annot_id:
                    print("Could not find wing annotation for item %s"%item_id)
                    continue
                resp = gc.get("annotation/%s" % annot_id)
                elements = resp['annotation']['elements']
                # fix a bug with the way arrows are saved.
                # conversion from angle to points is wrong.
                for e in elements:
                    if e['type'] == 'arrow':
                        pt0 = e['points'][0]
                        pt1 = e['points'][1]
                        dy = pt1[1] - pt0[1]
                        pt1[1] = pt0[1] - dy
                annot_dict[item_id] = elements

            center = arrow['center']
            radius = arrow['length']/2
            # find the wing base that is within this circle.
            for e in elements:
                if e['type'] == 'arrow':
                    # the wing base.
                    pt1 = e['points'][1]
                    dist = distance(center, pt1)
                    if dist < radius:
                        arrow['wing'] = e
                        break;
            if not 'wing' in arrow:
                print("-----------------------------------------------------")
                print("Could not find wing for arrow %d in image %s"%(count, arrow['item_id']))
                pprint(arrow['points'])
            
    return arrow_image_sets


# this only works with png files.
def download_files_from_item_id(image_id, dir_path, gc=None):
    file_path = None
    gc = get_gc()
    resp = gc.get("item/%s/files"%image_id)
    png_id = None
    for file_info in resp:
        file_id = file_info['_id']
        file_name = file_info['name']
        print(file_name)
        url = GIRDER_URL + "/api/v1/file/" + file_id + "/download"
        if sys.version_info[0] < 3:
            req = urllib2.Request(url)
            req.add_header('Girder-Token', gc.token)
            try:
                resp = urllib2.urlopen(req)
                #image = np.asarray(bytearray(resp.read()), dtype="uint8")
                file_path = os.path.join(dir_path, file_name)
                with open(file_path, 'wb+') as f:
                    f.write(resp.read())
            except urllib2.HTTPError as err:
                if err.code == 400:
                    print("Bad request!")
                elif err.code == 404:
                    print("Page not found!")
                elif err.code == 403:
                    print("Access denied!")
                else:
                    print("Something happened! Error code %d" % err.code)
        else:
            req = urllib.request.Request(url)
            req.add_header('Girder-Token', gc.token)
            try:
                resp = urllib.request.urlopen(req)
                #image = np.asarray(bytearray(resp.read()), dtype="uint8")
                file_path = os.path.join(dir_path, file_name)
                with open(file_path, 'wb+') as f:
                    f.write(resp.read())
            except urllib.error.HTTPError as err:
                if err.code == 400:
                    print("Bad request!")
                elif err.code == 404:
                    print("Page not found!")
                elif err.code == 403:
                    print("Access denied!")
                else:
                    print("Something happened! Error code %d" % err.code)

    # assumes only one file.
    return file_path



# Download the images from Girder (cache locally) to speed up initialization (loading)
# one time use.
def download_images_from_folder(gc, folder_id):
    #folder_id = "5acbb7423f24e537140b66aa"
    resp = gc.get("item?folderId=%s&limit=5000"%folder_id)
    dir_path = "/home/local/KHQ/charles.law/data/view/train"
    for item in resp:
        image_id = item['_id']
        #print(item['name'])
        file_path = download_files_from_item_id(image_id, dir_path, gc)



def get_item_from_description(description, gc=None):
    gc = get_gc()

    resp = gc.get('item?text=%s'%description)
    for item in resp:
        if item['description'] == description:
            return item
    return None



if __name__ == '__main__':
    """
    file_to_heatmap("5915da6add98b578723a09cb", "masks.png")
    file_to_heatmap("5915da6add98b578723a09cb", "error_map.png")
    file_to_heatmap("5915da6add98b578723a09cb", "error_map2.png")
    file_to_heatmap("5915da6add98b578723a09cb", "error_map3.png")
    file_to_heatmap("5915da6add98b578723a09cb", "error_map4.png")
    #file_to_heatmap("5915da6add98b578723a09cb", "prediction1.png")
    #file_to_heatmap("5915da6add98b578723a09cb", "prediction2.png")
    file_to_heatmap("5915da6add98b578723a09cb", "prediction3.png")
    file_to_heatmap("5915da6add98b578723a09cb", "prediction4.png")
    file_to_heatmap("5915da6add98b578723a09cb", "prediction6.png")
    """

    map = get_heatmap_image("5915da6add98b578723a09cb", 'masks.png')

    region = [int(map.region[2]*0.49), int(map.region[3]*0.42),
              int(map.region[2]*0.75), int(map.region[3]*0.57)]
    map.zero(region, outside=False)

    heatmap = Heatmap(map.image)
    heatmap.save_to_girder("5915da6add98b578723a09cb", "test_mask.png")

