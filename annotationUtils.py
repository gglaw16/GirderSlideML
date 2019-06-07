import Tkinter as tk
from pprint import pprint
import sys
import json
import os
import math
import cv2
import numpy as np
import scipy.misc
from PIL import Image
import pdb


# Common methods from all allotation scripts.  It expects a global variable called VIEWER to
# be created by the main script.  THis is for Tk/GUI callback functions.
# TODO:
# Iterators should have a superclass that handles file iteration.
# TODO: save to disk after "dots" delete.


# unique, double fuselage: ./train/Shenyang/3857_12_3452_1522_20170120_f863025f-abfa-4fba-98dc-307911f05f09.json



def npToPhoto(arr):
    size = arr.shape[0:2]
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
    im = Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)
    photo = tk.PhotoImage(im)
    return photo


def npToPhoto2(chip):
    im=Image.frombytes('L', (chip.shape[1],chip.shape[0]),
                       chip.astype('b').tostring())
    photo = tk.PhotoImage(im)
    return photo


def npToPhoto3(chip):
    scipy.misc.imsave("/tmp/tmp.png", chip)
    photo = tk.PhotoImage(file="/tmp/tmp.png")
    return photo



# widget api
#render()
#reset()
#record_annotation()



class Cursor:
    def __init__(self, canvas):
        self.canvas = canvas
        self.lines = None

    def draw(self, x, y):
        r = 20
        if self.lines is None:
            self.lines = [self.canvas.create_line(x-r, y, x+r, y, fill='#00FF00'), \
                          self.canvas.create_line(x, y-r, x, y+r, fill='green')]
        else:
            self.canvas.coords(self.lines[0], (x-r, y, x+r, y))
            self.canvas.coords(self.lines[1], (x, y-r, x, y+r))


class Arrow:
    """
    Renders a single arrow.
    """
    def __init__(self, viewer):
        """
        viewer: holds the canvas where the arrow is drawn.
        """
        self.viewer = viewer
        self.canvas = viewer.canvas
        self.modified = False
        self.line=None
        self.circle=None
        # tip
        self.x1 = 0
        self.y1 = 0
        # tail
        self.x2 = 0
        self.y2 = 0
        
    
    def contains_id(self, shape_id):
        return shape_id == self.line or shape_id == self.circle
            
        
    def highlight(self, flag):
        if flag:
            self.canvas.itemconfigure(self.line,fill="yellow")
            self.canvas.itemconfigure(self.circle,fill="yellow")
        else:
            self.canvas.itemconfigure(self.line,fill="red")
            self.canvas.itemconfigure(self.circle,fill="#1f1")
                
        
    def reset(self):
        self.modified = False
        if self.line:
            self.canvas.delete(self.line)
            self.line = None
            self.canvas.delete(self.circle)
            self.circle = None

            
    def render(self, plane, field_name):
        self.reset()

        if field_name in plane:
            self.plane = plane
            self.field_name = field_name
            e = plane[field_name]
            pt1 = self.viewer.world_point_to_chip((e[0], e[1]))
            pt2 = self.viewer.world_point_to_chip((e[2], e[3]))
            self.modified = False
            head_size = 2
            # head
            self.x1 = pt1[0]
            self.y1 = pt1[1]
            # tail
            self.x2 = pt2[0]
            self.y2 = pt2[1]
            if self.line is None:
                self.line = self.canvas.create_line(self.x1, self.y1, self.x2, self.y2, fill='red')
                self.circle = self.canvas.create_oval(self.x1-head_size, self.y1-head_size,
                                                      self.x1+head_size, self.y1+head_size,
                                                      outline="#f11", fill="#1f1", width=1)
            else:
                self.canvas.coords(self.circle, (self.x1-head_size, self.y1-head_size,
                                                 self.x1+head_size, self.y1+head_size))
                self.canvas.coords(self.line, (self.x1, self.y1, self.x2, self.y2))


    def handle_button_motion(self, event):
        if not self.plane:
            return
        field = self.plane[self.field_name]

        head_size = 2
        x = event.x
        y = event.y
        d1 = (self.x1-x)*(self.x1-x) + (self.y1-y)*(self.y1-y)
        d2 = (self.x2-x)*(self.x2-x) + (self.y2-y)*(self.y2-y)
        if d1 < d2:
            self.x1 = x
            self.y1 = y
            self.canvas.coords(self.circle, (x-head_size, y-head_size, x+head_size, y+head_size))
            # Update the plane annotation
            pt = self.viewer.chip_point_to_world((x, y))
            field[0] = pt[0]
            field[1] = pt[1]
        else:
            self.x2 = x
            self.y2 = y
            # Update the plane annotation
            pt = self.viewer.chip_point_to_world((x, y))
            field[2] = pt[0]
            field[3] = pt[1]
        # I could just call render again....
        self.canvas.coords(self.line, (self.x1, self.y1, self.x2, self.y2))

    
    def handle_mouse_wheel(self, event):
        return

            
            
# I do not know if this buys us anything over the tk shape.
class Rectangle:
    """
    Renders a single rectangle.
    """
    def __init__(self, viewer):
        self.viewer = viewer
        self.canvas = viewer.canvas
        self.modified = False
        self.lines=[]
        self.center = [300,300]
        self.radius = 150


    def contains_id(self, shape_id):
        return shape_id in self.lines
            
        
    def highlight(self, flag):
        if flag:
            for line in self.lines:
                self.canvas.itemconfigure(line,fill="yellow")
        else:
            for line in self.lines:
                self.canvas.itemconfigure(line,fill="red")
                
        
    def render(self, plane, field_name):
        self.reset()

        if field_name in plane:
            self.plane = plane
            self.field_name = field_name
            e = plane[field_name]
            pt1 = self.viewer.world_point_to_chip((e[0], e[1]))
            pt2 = self.viewer.world_point_to_chip((e[2], e[3]))
            self.modified = False
            self.center = (int((pt1[0]+pt2[0])/2), int((pt1[1]+pt2[1])/2))
            dx = (pt1[0]-pt2[0])
            dy = (pt1[1]-pt2[1])
            self.radius = math.sqrt(dx*dx + dy*dy) / 2.0
            self.update()
            self.modified = False        


    def update(self):
        self.modified = True
        c = self.center
        r = self.radius
        rh = int(r/2)
        if len(self.lines) == 0:
            # top. right, botto, left
            l = self.canvas.create_line(c[0]-r,c[1]-r, c[0]+r,c[1]-r, fill='red')
            self.lines.append(l)
            l = self.canvas.create_line(c[0]+r,c[1]-r, c[0]+r,c[1]+r, fill='red')
            self.lines.append(l)
            l = self.canvas.create_line(c[0]+r,c[1]+r, c[0]-r,c[1]+r, fill='red')
            self.lines.append(l)
            l = self.canvas.create_line(c[0]-r,c[1]+r, c[0]-r,c[1]-r, fill='red')
            self.lines.append(l)
            # crosshairs in center
            l = self.canvas.create_line(c[0]-rh,c[1], c[0]+rh,c[1], fill='blue')
            self.lines.append(l)
            l = self.canvas.create_line(c[0],c[1]-rh, c[0],c[1]+rh, fill='blue')
            self.lines.append(l)
        else:
            # top. right, bottom, left
            self.canvas.coords(self.lines[0], (c[0]-r,c[1]-r, c[0]+r,c[1]-r))
            self.canvas.coords(self.lines[1], (c[0]+r,c[1]-r, c[0]+r,c[1]+r))
            self.canvas.coords(self.lines[2], (c[0]+r,c[1]+r, c[0]-r,c[1]+r))
            self.canvas.coords(self.lines[3], (c[0]-r,c[1]+r, c[0]-r,c[1]-r))
            # crosshairs in center
            self.canvas.coords(self.lines[4], (c[0]-rh,c[1], c[0]+rh,c[1]))
            self.canvas.coords(self.lines[5], (c[0],c[1]-rh, c[0],c[1]+rh))
            
        
    def reset(self):
        self.modified = False
        for line in self.lines:
            self.canvas.delete(line)
        self.lines = []

        
    def handle_button_motion(self, event):
        if self.plane:
            pts = self.plane[self.field_name]
            center = ((pts[0]+pts[2])*0.5, (pts[1]+pts[3])*0.5)
            pt = self.viewer.chip_point_to_world((event.x, event.y))
            dx = pt[0] - center[0]
            dy = pt[1] - center[1]
            pts[0] += dx
            pts[1] += dy
            pts[2] += dx
            pts[3] += dy
            self.render(self.plane, self.field_name)


    def handle_mouse_wheel(self, event):
        if self.plane:
            pts = self.plane[self.field_name]
            center = ((pts[0]+pts[2])*0.5, (pts[1]+pts[3])*0.5)
            half_width = abs(pts[2]-pts[0])*0.5
            half_height = abs(pts[3]-pts[1])*0.5
            if event.num == 5 or event.delta == -120:
                half_width = half_width * 0.9
                half_height = half_height * 0.9
            if event.num == 4 or event.delta == 120:
                half_width = half_width / 0.9
                half_height = half_height / 0.9
            pts[0] = center[0] - half_width
            pts[1] = center[1] - half_height
            pts[2] = center[0] + half_width
            pts[3] = center[1] + half_height
            self.render(self.plane, self.field_name)

            
        
class Dots:
    """
    Renders a dot for every plane/detection in the viewers.annotations.
    New dots/detections can be added my clicking in the viewer
    """
    def __init__(self, viewer):
        """
        viewer: holds the canvas where the arrow is drawn,
                and has the annotations that places the dots.
        """
        self.viewer = viewer
        self.canvas = viewer.canvas
        self.modified = False
        self.dots = []
        self.radius = 10 
        self.canvas.bind('<Button-1>', self.handle_button)
        self.shapes = []
        self.picked_shape = None
        

    def reset(self):
        self.modified = False
        for dot in self.dots:
            self.canvas.delete(dot)
        self.dots = []

        for shape in self.shapes:
            shape.reset()
        self.shapes = []
            
        
    def get_dot_from_plane(self, plane):
        #center = plane['detection']['center']
        #size = plane['detection']['rf_size'] * plane['detection']['spacing']
        #radius = int(size / 2)
        if 'nose-tail' in plane:
            pts = plane['nose-tail']
            dx = pts[2]-pts[0]
            dy = pts[3]-pts[1]
            center = [(pts[2]+pts[0])*0.5, (pts[3]+pts[1])*0.5]
            size = math.sqrt(dx*dx + dy*dy)
        elif 'bbox' in plane:
            bbox = plane['bbox']
            center = [(bbox[2]+bbox[0])*0.5, (bbox[3]+bbox[1])*0.5]
            left = min(bbox[0], bbox[2])
            top = min(bbox[1], bbox[3])
            right = max(bbox[0], bbox[2])
            bottom = max(bbox[1], bbox[3])
            size = max(abs(right-left), abs(top-bottom))
        else:
            return None, None
        
        radius = int(size / 2)
        return center, radius

        
    # I am choosing to only have dots in the viewer which are visible.
    # I am deleting all dots and recreating them.
    # I could try to reuse old dots in the future.
    def render(self):
        self.reset()
        annotations = self.viewer.get_annotations()
        planes = annotations.get_elements()
        if planes == None:
            return
        for plane in planes:
            # TODO: FInd a better API to clip shapes.
            center, radius = self.get_dot_from_plane(plane)
            if center == None:
                continue
            pt = self.viewer.world_point_to_chip(center)
            radius = radius * self.viewer.get_scale()
            if pt[0]+radius > 0 and pt[0]-radius < self.viewer.canvas_width and \
               pt[1]+radius > 0 and pt[1]-radius < self.viewer.canvas_height:
                if 'nose-tail' in plane:
                    arrow = Arrow(self.viewer)
                    self.shapes.append(arrow)
                    arrow.render(plane, 'nose-tail')
                elif 'bbox' in plane:
                    rect = Rectangle(self.viewer)
                    self.shapes.append(rect)
                    rect.render(plane, 'bbox')

            #    dot = self.canvas.create_oval(pt[0]-radius, pt[1]-radius,
            #                                  pt[0]+radius, pt[1]+radius,
            #                                  outline="#f11", width=2)
            #    self.dots.append(dot)
            

    def record_annotation(self):
        # Dots are recorded when they are created.
        return

    
    def handle_key(self, event):
        if event.keycode == 107 or event.keycode == 119:
            if self.picked_shape:
                shape = self.picked_shape
                del shape.plane[shape.field_name]
                self.render()
        if event.keycode == 38:
            # a
            self.add_arrow(event)
        if event.keycode == 39:
            # s
            self.add_square(event)
        

    def add_arrow(self, event):
        x,y = self.viewer.chip_point_to_world((event.x, event.y))
        annotations = self.viewer.get_anntoations()
        plane = {'nose-tail': [x, y, x+100, y]}
        annotations.get_elements().append(plane)
        arrow = Arrow(self.viewer)
        self.shapes.append(arrow)
        arrow.render(plane, 'nose-tail')
            
                
    def add_square(self, event):
        x,y = self.viewer.chip_point_to_world((event.x, event.y))
        annotations = self.viewer.get_anntoations()
        plane = {'bbox': [x-50, y-50, x+50, y+50]}
        annotations.get_elements().append(plane)
        square = Rectangle(self.viewer)
        self.shapes.append(square)
        square.render(plane, 'bbox')
        self.set_picked_shape(square)

            
    def handle_button_motion(self, event):
        if self.picked_shape:
            self.picked_shape.handle_button_motion(event)

                    
    def handle_mouse_wheel(self, event):
        if self.picked_shape:
            self.picked_shape.handle_mouse_wheel(event)
        
        
    def handle_button(self, event):
        self.set_picked_shape(self.point_to_shape((event.x, event.y)))

                
    def point_to_shape(self, pt):
        shape_id  = self.canvas.find_closest(pt[0], pt[1], halo = 5)[0]
        for shape in self.shapes:
            if shape.contains_id(shape_id):
                return shape
        return None
        
                       
    def set_picked_shape(self, shape):
        if self.picked_shape:
            self.picked_shape.highlight(False)
        self.picked_shape = shape;
        if shape:
            shape.highlight(True)



                
    #    pt = self.viewer.chip_point_to_world((event.x, event.y))
    #    radius = self.radius
    #    dot = self.canvas.create_oval(pt[0]-radius, pt[1]-radius,
    #                                  pt[0]+radius, pt[1]+radius,
    #                                  outline="#f11", width=2)
    #    self.dots.append(dot)
    #    # add the detection to the annotation at the same time we add the dot.
    #    spacing = 1.0 / self.viewer.get_scale()
    #    world_pt = self.viewer.chip_point_to_world(pt)
    #    plane = {'detection': {'score': 1.0,
    #                           'spacing': spacing,
    #                           'center': world_pt,
    #                           'rf_size': self.radius}}
    #    annotations = self.viewer.get_annotations()
    #    annotations['planes'].append(plane)

        
        

class ArrowWidget:
    """
    Renders a single arrow representing the current plane in the iterator.
    """
    def __init__(self, viewer, iterator, field_name):
        """
        viewer: holds the canvas where the arrow is drawn.
        iterator: source of the plane (current) which will be rendere.
        field_name: The field of the plane that will be drawn.
        """
        self.iterator = iterator
        self.field_name = field_name
        self.viewer = viewer
        self.canvas = viewer.canvas
        self.modified = False
        self.line=None
        # tip
        self.x1 = 0
        self.y1 = 0
        # tail
        self.x2 = 0
        self.y2 = 0

        self.canvas.bind('<Button-1>', self.handle_button)
        self.canvas.bind('<B1-Motion>', self.handle_button)

        
    def handle_button(self, event):
        self.draw(event.x, event.y)

        
    def handle_button_motion(self, event):
        return
        
        
    def handle_mouse_wheel(self, event):
        return
        
        
    def record_annotation(self):
        plane = self.iterator.get_current()
        if plane is None:
            return
        if not 'tags' in plane:
            plane['tags'] = []

        # TODO:  Clean this up.  Too many cases specific to user annotation task.
        # Record the arrow, square to the annotation.
        if self.modified:
            pt1 = self.viewer.chip_point_to_world((self.x1, self.y1))
            pt2 = self.viewer.chip_point_to_world((self.x2, self.y2))
            plane[self.field_name] = [pt1[0], pt1[1], pt2[0], pt2[1]]
            if 'tags' in plane and 'negative' in plane['tags']:
                plane['tags'].remove('negative')

    
    def reset(self):
        self.modified = False
        if self.line:
            self.canvas.delete(self.line)
            self.line = None
            self.canvas.delete(self.circle)
            self.circle = None

    def render(self):
        plane = self.iterator.get_current()
        self.reset()

        if self.field_name in plane:
            e = plane[self.field_name]
            pt1 = self.viewer.world_point_to_chip((e[0], e[1]))
            pt2 = self.viewer.world_point_to_chip((e[2], e[3]))
            self.modified = False
            head_size = 2
            # head
            self.x1 = pt1[0]
            self.y1 = pt1[1]
            # tail
            self.x2 = pt2[0]
            self.y2 = pt2[1]
            if self.line is None:
                self.line = self.canvas.create_line(self.x1, self.y1, self.x2, self.y2, fill='red')
                self.circle = self.canvas.create_oval(self.x1-head_size, self.y1-head_size,
                                                      self.x1+head_size, self.y1+head_size,
                                                      outline="#f11", fill="#1f1", width=1)
            else:
                self.canvas.coords(self.circle, (self.x1-head_size, self.y1-head_size,
                                                 self.x1+head_size, self.y1+head_size))
                self.canvas.coords(self.line, (self.x1, self.y1, self.x2, self.y2))

            
    # This is for interaction.
    def draw(self, x, y):
        """
        position the tip or root, which ever is closer, at the point.
        """
        self.modified = True
        head_size = 2
        if self.line is None:
            # head
            self.x1 = x
            self.y1 = y
            # tail
            self.x2 = x
            self.y2 = y
            self.line = self.canvas.create_line(self.x1, self.y1, self.x2, self.y2, fill='red')

            self.circle = self.canvas.create_oval(x-head_size, y-head_size, x+head_size, y+head_size,
                                                  outline="#f11", fill="#1f1", width=1)
        else:
            d1 = (self.x1-x)*(self.x1-x) + (self.y1-y)*(self.y1-y)
            d2 = (self.x2-x)*(self.x2-x) + (self.y2-y)*(self.y2-y)
            if d1 < d2:
                self.x1 = x
                self.y1 = y
                self.canvas.coords(self.circle, (x-head_size, y-head_size, x+head_size, y+head_size))
            else:
                self.x2 = x
                self.y2 = y
            self.canvas.coords(self.line, (self.x1, self.y1, self.x2, self.y2))

    def handle_key(self, event):
        nop = event
        


class SquareWidget:
    """
    Renders a single square representing the current plane in the iterator.
    """
    def __init__(self, viewer, iterator, field_name):
        self.iterator = iterator
        self.field_name = field_name
        self.viewer = viewer
        self.canvas = viewer.canvas
        self.modified = False
        self.lines=[]
        self.center = [300,300]
        self.radius = 150

        canvas.bind('<Button-4>', self.update_size)
        canvas.bind('<Button-5>', self.update_size)
        #canvas.bind('<MouseWheel>', self.update_size)
        canvas.bind('<Button-1>', self.update_center)
        canvas.bind('<B1-Motion>', self.update_center)

        
    def handle_mouse_wheel(self, event):
        self.update_size(event)
        
        
    def render(self):
        plane = self.iterator.get_current()
        self.reset()

        if self.field_name in plane:
            e = plane[self.field_name]
            pt1 = self.viewer.world_point_to_chip((e[0], e[1]))
            pt2 = self.viewer.world_point_to_chip((e[2], e[3]))
            self.modified = False
            self.center = (int((pt1[0]+pt2[0])/2), int((pt1[1]+pt2[1])/2))
            dx = (pt1[0]-pt2[0])
            dy = (pt1[1]-pt2[1])
            self.radius = math.sqrt(dx*dx + dy*dy) / 2.0
            self.update()
            self.modified = False        


    def update_center(self, event):
        self.canvas.focus_set()
        self.center = (event.x, event.y)
        print(self.center)
        self.update()

        
    def update_size(self, event):
        # respond to Linux or Windows wheel event
        if event.num == 5 or event.delta == -120:
            self.radius -= 5
        if event.num == 4 or event.delta == 120:
            self.radius += 5
        self.update()

        
    def update(self):
        self.modified = True
        c = self.center
        r = self.radius
        rh = int(r/2)
        if len(self.lines) == 0:
            # top. right, botto, left
            l = self.canvas.create_line(c[0]-r,c[1]-r, c[0]+r,c[1]-r, fill='red')
            self.lines.append(l)
            l = self.canvas.create_line(c[0]+r,c[1]-r, c[0]+r,c[1]+r, fill='red')
            self.lines.append(l)
            l = self.canvas.create_line(c[0]+r,c[1]+r, c[0]-r,c[1]+r, fill='red')
            self.lines.append(l)
            l = self.canvas.create_line(c[0]-r,c[1]+r, c[0]-r,c[1]-r, fill='red')
            self.lines.append(l)
            # crosshairs in center
            l = self.canvas.create_line(c[0]-rh,c[1], c[0]+rh,c[1], fill='blue')
            self.lines.append(l)
            l = self.canvas.create_line(c[0],c[1]-rh, c[0],c[1]+rh, fill='blue')
            self.lines.append(l)
        else:
            # top. right, bottom, left
            self.canvas.coords(self.lines[0], (c[0]-r,c[1]-r, c[0]+r,c[1]-r))
            self.canvas.coords(self.lines[1], (c[0]+r,c[1]-r, c[0]+r,c[1]+r))
            self.canvas.coords(self.lines[2], (c[0]+r,c[1]+r, c[0]-r,c[1]+r))
            self.canvas.coords(self.lines[3], (c[0]-r,c[1]+r, c[0]-r,c[1]-r))
            # crosshairs in center
            self.canvas.coords(self.lines[4], (c[0]-rh,c[1], c[0]+rh,c[1]))
            self.canvas.coords(self.lines[5], (c[0],c[1]-rh, c[0],c[1]+rh))
            
        
    def record_annotation(self):
        plane = self.iterator.get_current()
        if plane is None:
            return
        if not 'tags' in plane:
            plane['tags'] = []

        # TODO:  Clean this up.  Too many cases specific to user annotation task.
        # Record the arrow, square to the annotation.
        if self.modified:
            c = self.center
            r = self.radius
            pt1 = self.viewer.chip_point_to_world((c[0]-r, c[1]-r))
            pt2 = self.viewer.chip_point_to_world((c[0]+r, c[1]+r))
            plane[self.field_name] = [pt1[0], pt1[1], pt2[0], pt2[1]]
            if 'tags' in plane and 'negative' in plane['tags']:
                plane['tags'].remove('negative')

    
    def reset(self):
        self.modified = False
        for line in self.lines:
            self.canvas.delete(line)
        self.lines = []


    def handle_key(self, event):
        return
        
    def handle_button_motion(self, event):
        return

        


class Annotation:
    """ 
    Manages both the json annotion and the satellite image.
    It does not keep track of which annotation is "current" (yet?)
    """
    def __init__(self, json_filename, chip_annot_field=None):
        """
        chip_annot_field: the annotation field used to sample the chip
          scale and orientation. If it is an arrow, the chips are rotated 
          to a common orientation when retured.
 
        targetName: The new arrow annotation being generated.
        """
        print(json_filename)
        self.json_filename = json_filename
        
        self.sat_image = None
        self.chip_annot_field = chip_annot_field
        self.padsize = 300
        
        with open(self.json_filename) as f:
            self.data = json.load(f)

        if not 'image_dir' in self.data:
            self.data['image_dir'] = ""
        self.image_filepath = os.path.join(self.data['image_dir'], self.data['image_filename'])
                                                     
        # TODO: check to see if we can actualy get to the image.  Add some logic to deal with
        # absolute and relative paths.


    def get_elements(self):
        return self.data['planes']

        
    def get_sat_image(self):
        """
        Load the large satellite image on demand.
        """
        if self.sat_image is None:
            self.sat_image = cv2.imread(self.image_filepath)
            # scipy will not open large images. Warns of a decompression bomb.
            self.sat_image = cv2.cvtColor(self.sat_image, cv2.COLOR_BGR2RGB)
        return self.sat_image

    
    def get_overlayed_image(self, feild='dot'):
        """
        Returns the satelite image with annotation drawn on top in red.
        """
        img = self.get_sat_image()
        overlay = img.copy()

        for plane in self.data.planes:
            if field == 'dot':
                dot = plane['dot']
                cv2.circle(overlay, dot['center'], 5, (0,0,255), -1)
        # blend with the original:
        opacity = 0.4
        cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

        return img
        
    
    def save(self):
        with open(self.json_filename, "w") as write_file:
            json.dump(self.data, write_file)

            
    def plane_to_chip_params(self, plane, chip_height):
        """
        Get the sample pameters (cx, cy, rotation, scale) from the chip annotation field.
        (cx, cy) is the center of the chip in sat_image coordinates.
        rotation ....
        scale: 1/spacing
        """
        annot = plane[self.chip_annot_field]
        
        if isinstance(annot, list):
            # So far, all anno fields have two points in an array [x1,y1,x2,y2]
            # compute the center and magnitude of the annotation
            cx = int((annot[0] + annot[2]) * 0.5)
            cy = int((annot[1] + annot[3]) * 0.5)
            dx = annot[0] - annot[2]
            dy = annot[1] - annot[3]
            mag = math.sqrt(dx*dx + dy*dy)
            scale = chip_height / (2*mag)
            # Get the matrix necessary to normalize rotation and scale. 
            if self.chip_annot_field == 'bbox':
                rotation = 0
            else:
                rotation = 90 + math.atan2(dy/mag, dx/mag) * 180.0 / math.pi
            return (cx,cy),rotation,scale
                
        if isinstance(annot, dict) and 'center' in annot:
            cx, cy = annot['center']
            scale = chip_height / (annot['rf_size'] * annot['spacing'])
            rotation = 0
            return (cx,cy),rotation,scale

                            
    # TODO: Change "element" to some better name.
    def get_chip(self, center, rotation, scale, chip_height):
        """
        center: center point of the chip in image coordinates.
        """
        sat_image = self.get_sat_image()
        width = chip_height

        M = cv2.getRotationMatrix2D(tuple(center), rotation, scale)
        # Shift to put 'center' in the center of the chip being returned.
        image_center = (sat_image.shape[1]/2, sat_image.shape[0]/1.1)
        chip_center = (width/2, chip_height/2)    
        offset = [center[0] - chip_center[0], center[1] - chip_center[1]]
        M[0][2] -= offset[0];
        M[1][2] -= offset[1];

        # Now resample the chip.
        pad_vector = (127,127,127)
        chip = cv2.warpAffine(sat_image, M, (width, chip_height), \
                              borderMode=cv2.BORDER_CONSTANT, \
                              borderValue=pad_vector, flags=cv2.INTER_LINEAR)

        # We need these matrixes to convert arrows back to annotation
        # in the sat_image coordinate system.
        self.world_to_chip = np.zeros((3,3))
        self.world_to_chip[0:2,0:3] = M
        self.world_to_chip[2,2] = 1.0
        self.chip_to_world = np.linalg.inv(self.world_to_chip)

        return chip

    
    def world_point_to_chip(self, wpt):
        M = self.world_to_chip
        x = wpt[0] * M[0,0] + wpt[1] * M[0,1] + M[0,2]
        y = wpt[0] * M[1,0] + wpt[1] * M[1,1] + M[1,2]
        return (x,y)

    def chip_point_to_world(self, wpt):
        M = self.chip_to_world
        x = wpt[0] * M[0,0] + wpt[1] * M[0,1] + M[0,2]
        y = wpt[0] * M[1,0] + wpt[1] * M[1,1] + M[1,2]
        return (x,y)




    

class Viewer:
    def __init__(self, title="", cursor=True):
        """
        """
        self.canvas_width = 600
        self.canvas_height = 600
        self.scale = 1.0
        self.TK_IMAGE = None
        self.photo = None
        self.text = None

        self.iterator = None
        self.annotations = None

        self.root = tk.Tk()
        self.root.title(title)
        self.frame = tk.Frame(self.root, width=self.canvas_width, height=self.canvas_height)
        self.frame.pack()
        self.canvas = tk.Canvas(self.frame, width=self.canvas_width,height=self.canvas_height)
        self.canvas.place(x=-2,y=-2)
        self.cursor = None
        if cursor:
            self.cursor = Cursor(self.canvas)
            self.canvas.bind('<Motion>', self.move_cursor)
        self.canvas.bind('<B1-Motion>', self.handle_button_motion)
        self.canvas.bind('<Key>', self.handle_key)
        self.canvas.bind('<Button-3>', self.handle_button3)
        #self.canvas.bind('<MouseWheel>', self.handle_mouse_wheel)
        self.canvas.bind('<Button-4>', self.handle_mouse_wheel)
        self.canvas.bind('<Button-5>', self.handle_mouse_wheel)

        self.widget = None

        
    def handle_button_motion(self, event):
        if self.cursor:
            self.move_cursor(event)
        if self.widget:
            self.widget.handle_button_motion(event)

            
    def handle_mouse_wheel(self, event):
        if self.widget:
            self.widget.handle_mouse_wheel(event)

            
    def set_annotations(self, annotations):
        self.annotations= annotations

        
    def get_annotations(self):
        return self.annotations

    
    def get_scale(self):
        return self.scale

    
    # Just because the annotation has the image file name, does not mean it should
    # manage the camera transformations.
    # TODO: Change this.
    def world_point_to_chip(self, wpt):
        return self.annotations.world_point_to_chip(wpt)

    
    def chip_point_to_world(self, wpt):
        return self.annotations.chip_point_to_world(wpt)

        
    def set_iterator(self, iterator):
        self.iterator = iterator

        
    def set_widget(self, widget):
        self.widget = widget

        
    def get_canvas_shape(self):
        return (self.canvas_height, self.canvas_width)

        
    def load_chip(self, chip):
        self.photo = npToPhoto3(chip)
        if self.TK_IMAGE is None:
            self.TK_IMAGE = self.canvas.create_image(0,0, anchor=tk.NW, image=self.photo)
        else:
            self.canvas.itemconfig(self.TK_IMAGE, image = self.photo)
        

    def render(self, center, rotation, scale):
        self.scale
        chip = self.annotations.get_chip(center, rotation, scale, self.canvas_height)        
        self.load_chip(chip)
        if self.widget:
            self.widget.render()
        
                
    def move_cursor(self, event):
        # Allows canvas to get key events.
        self.canvas.focus_set()
        self.cursor.draw(event.x, event.y)


    def handle_key(self, event):
        pprint(event.keycode)
        if event.keycode == 9:
            # esc key
            sys.exit(0)
        if event.keycode == 40:
            # d key (is u, ctrl-u or ctrl-z for undo better?)
            if self.widget:
                self.widget.reset()
            return
        if self.widget:
            self.widget.handle_key(event)
        if self.iterator:
            self.iterator.handle_key(event)


    def handle_button3(self, event):
        if self.iterator:
            self.iterator.handle_button3(event)

            
    def record_annotation(self):
        """
        """
        if self.widget:
            self.widget.record_annotation()
        # TODO: Figure out how to mark detection as a negative if user skipped it.
        #if self.iteration_name == 'detection':
        #    if not 'negative' in plane['tags']:
        #        plane['tags'].append('negative')
            
        # Save the annotation back to disk.    
        self.annotations.save()
        
        
    def start(self):
        self.root.mainloop()




class AnnotationIterator:
    def __init__(self, viewer, json_filenames, iteration_name, skip_name=None):
        """
        iteration_name:  elements wthput this type of annotation will be skipped.
        skip_name: elements with this type of annotation will be skipped.
        """
        self.iteration_name = iteration_name
        self.skip_name = skip_name
        self.iteration_index = -1
        self.planes = []
        # Annotation object
        self.annotations = None
        self.json_filenames = json_filenames
        self.viewer = viewer

        
    def handle_button_motion(self, event):
        x = event.x
        

    def handle_key(self, event):
        if event.keycode == 102 or event.keycode == 36:  #114?
            # right arrow or enter key
            self.next()
        if event.keycode == 100:
            # left arrow or enter key
            self.previous()
        if event.keycode == 107:   #119?
            # delete key
            self.iterator.delete_current()
            self.next()

            
    def handle_button3(self, event):
        self.next()
            

    def set_annotations(self, annotations):
        """ 
        annotations: object with annotion elements already loaded.
        """
        self.viewer.set_annotations(annotations)

        if self.iteration_name is None:
            self.iteration_index = -1
            self.planes = []
            return;
        # uncomment to see them all. Otherwise it shows only incomplete annotation.
        #skip_name = None
        self.annotations = annotations
        # Filter the annotation elements that we will be iterating over.
        self.planes = []
        elements = annotations.get_elements()
        pdb.set_trace()
        for plane in elements:
            # I am planning to use tags for negaitve, and classification.
            if not 'tags' in plane:
                plane['tags'] = []
            if self.iteration_name in plane and not 'negative' in plane['tags']:
                if self.skip_name is None or not self.skip_name in plane:
                    self.planes.append(plane)
        self.iteration_index = -1

        
    def get_length(self):
        if self.planes:
            return len(self.planes)
        return 0

    
    def get_current(self):
        if self.planes and self.iteration_index >=0 and self.iteration_index < len(self.planes):
            return self.planes[self.iteration_index]
        return None
        

    def delete_current():
        annot = self.iterator.get_current()
        pprint(annot)
        # first get rid of the iteration field.  It must be bad.
        del annot[self.iteration_name]
        # If there are no other coordinate fields, get rid of the whole anntoation element.
        remove = True
        for field in annot:
            if isinstance(field, list) or isinstance(field, tuple):
                remove = False
        if remove:
            data_idx = self.annotations.data['planes'].index(annot)
            del self.annotations.data['planes'][data_idx]
            del self.planes[self.iterator.iteration_index]
            self.iterator.iteration_index -= 1


    def start(self):
        self.next_file()
        self.viewer.start()
        
            
    def next_file(self):
        if len(self.json_filenames) == 0:
            sys.exit(1)
        json_filename = self.json_filenames.pop()
        annotations = Annotation(json_filename, self.iteration_name)
        self.set_annotations(annotations)
        if self.get_length() == 0:
            # Empty annotation, advance to the next.
            self.next_file()
        else:
            self.goto(0)
            
            
    def next(self):
        self.viewer.record_annotation()
        self.iteration_index += 1
        if self.iteration_index >= len(self.planes):
            self.next_file()
        else:
            self.goto(self.iteration_index)
        

    def previous(self):
        self.viewer.record_annotation()
        if self.iteration_index == 0:
            print("Cannot go back")
            return
        self.iteration_index -= 1
        self.goto(self.iteration_index)
        

    # TODO: This is a mess.  FIx it.
    def goto(self, iteration_index):
        if iteration_index < 0 or iteration_index >= len(self.planes):
            print("element index out of range.")
            return

        self.iteration_index = iteration_index
        print("%d of %d"%(iteration_index, len(self.planes)))
        plane = self.get_current()

        # TODO: Fix this.
        canvas_height, canvas_width = self.viewer.get_canvas_shape()
        center,rotation,scale = self.viewer.annotations.plane_to_chip_params(plane, canvas_height)
        self.viewer.render(center, rotation, scale)


        # Update widgets that render info for the current plane.
        # I want to show the confidence for proofreading detections.
        if self.iteration_name == 'detection':
            score=str(round(plane['detection']['score'], 2))
            score_str = "score = %s"%score
            if self.viewer.text:
                self.viewer.canvas.itemconfig(self.text, text=score_str)
            else:
                self.viewer.text =  self.viewer.canvas.create_text(60,20,fill="darkblue",
                                                                   font="Times 20 italic bold",
                                                                   text=score_str)



class GridIterator:
    """
    Cover an entire image in a grid pattern with some overlap.
    """
    def __init__(self, viewer, json_filenames, spacing=1):
        """
        spacing: 1 is highest resolution, 2 is half resolution ...
        """
        self.grid_dims = (0, 0)
        self.grid_idx = [0,0]
        self.viewer = viewer
        self.spacing = spacing
        self.json_filenames = json_filenames

        
    def update(self):
        if self.grid_idx[0] < 0:
            self.grid_idx[0] = 0
            print("left edge")
            # beep
            sys.stdout.write('\a')
            sys.stdout.flush()
        if self.grid_idx[0] >= self.grid_dims[0]:
            self.grid_idx[0] = self.grid_dims[0]-1
            print("right edge")
            # beep
            sys.stdout.write('\a')
            sys.stdout.flush()
        if self.grid_idx[1] < 0:
            self.grid_idx[1] = 0
            print("top edge")
            # beep
            sys.stdout.write('\a')
            sys.stdout.flush()
        if self.grid_idx[1] >= self.grid_dims[1]:
            self.grid_idx[1] = self.grid_dims[1]-1
            print("bottom edge")
            # beep
            sys.stdout.write('\a')
            sys.stdout.flush()
        self.viewer.record_annotation()    
        self.viewer.load_chip(self.get_chip())
        
        
    def handle_key(self, event):
        if event.keycode == 36:  #114?
            # enter key
            self.next_file()
            return
        if event.keycode == 102:  #114?
            # right arrow
            self.grid_idx[0] += 1
            self.update()
            return
        if event.keycode == 100:
            # left arrow
            self.grid_idx[0] -= 1
            self.update()
            return
        if event.keycode == 98:
            # up arrow
            self.grid_idx[1] -= 1
            self.update()
            return
        if event.keycode == 104:
            # down arrow
            self.grid_idx[0] += 1
            self.update()
            return

        
    def handle_button3(self, event):
        return

    
    def handle_button_motion(self, event):
        return
    
    
    def set_annotations(self, annotations):
        self.viewer.set_annotations(annotations)

        view_height, view_width = self.viewer.get_canvas_shape()
        self.view_width = view_width
        self.view_height = view_height
        sat_image = annotations.get_sat_image()
        image_height, image_width, _ = sat_image.shape
        image_height = int(image_height / self.spacing)
        image_width = int(image_width / self.spacing)
        
        # find overlaps.  Start with 10%
        overlap_x = int(0.1 * view_width)
        overlap_y = int(0.1 * view_height)
        # compute the grid dimensions with the minimum overlap
        grid_dim_x = int(math.ceil(float(image_width - overlap_x) / (view_width - overlap_x))) 
        grid_dim_y = int(math.ceil(float(image_height - overlap_y) / (view_height - overlap_y))) 
        # Let the overlaps grow so the grid edges match the images.
        overlap_x  = int(float(grid_dim_x*view_width - image_width) / (grid_dim_x - 1))
        overlap_y  = int(float(grid_dim_y*view_height - image_height) / (grid_dim_y - 1))

        self.grid_dims = (grid_dim_x, grid_dim_y)
        self.overlaps = (overlap_x, overlap_y)
        self.iteration_index = None

        
    def get_length(self):
        return self.grid_dims[0] * self.grid_dims[1]

    
    def get_current(self):
        return None
        

    def goto(self, iteration_index):
        y_idx = int(iteration_index / self.grid_dims[0])
        x_idx = int(iteration_index - y_idx*self.grid_dims[0])

        if x_idx < self.grid_dims[0]:
            self.grid_idx = [x_idx, y_idx]
            self.viewer.load_chip(self.get_chip())


    def get_chip(self):
        left = self.grid_idx[0] * (self.view_width-self.overlaps[0])
        right = left + self.view_width
        top = self.grid_idx[1] * (self.view_height-self.overlaps[1])
        bottom = top + self.view_height
        # TODO: handle the image better. Annotations?
        sat_image = self.viewer.annotations.get_sat_image()
        return sat_image[top:bottom, left:right, :]
        

    def start(self):
        self.next_file()
        self.viewer.start()
        

    def next_file(self):
        if len(self.json_filenames) == 0:
            sys.exit(1)
        json_filename = self.json_filenames.pop()
        annotations = Annotation(json_filename, self.iteration_name)
        self.set_annotations(annotations)
        self.goto(0)

            


class ChipIterator:
    def __init__(self, viewer, json_chip_names):
        """
        """
        # Annotation object
        self.annotations = None
        self.chip_file_paths = json_chip_names
        self.viewer = viewer

        
    def handle_key(self, event):
        if event.keycode == 102 or event.keycode == 36:  #114?
            # right arrow or enter key
            self.next()
        #if event.keycode == 100:
        #    # left arrow or enter key
        #    self.previous()
        #if event.keycode == 107:   #119?
        #    # delete key
        #    self.iterator.delete_current()
        #    self.next()

            
    def handle_button3(self, event):
        self.next()

        
    def handle_button_motion(self, event):
        return

        
    #def get_length(self):
    #    if self.json_filenames:
    #        return len(self.json_filenames)
    #    return 0

    
    #def get_current(self):
    #    if self.planes and self.iteration_index >=0 and self.iteration_index < len(self.planes):
    #        return self.planes[self.iteration_index]
    #    return None
        

    #def delete_current():
    #    annot = self.iterator.get_current()
    #    pprint(annot)
    #    # first get rid of the iteration field.  It must be bad.
    #    del annot[self.iteration_name]
    #    # If there are no other coordinate fields, get rid of the whole anntoation element.
    #    remove = True
    #    for field in annot:
    #        if isinstance(field, list) or isinstance(field, tuple):
    #            remove = False
    #    if remove:
    #        data_idx = self.annotations.data['planes'].index(annot)
    #        del self.annotations.data['planes'][data_idx]
    #        del self.planes[self.iterator.iteration_index]
    #        self.iterator.iteration_index -= 1


    def start(self):
        self.next()
        self.viewer.start()
        
            
    def next(self):
        if self.annotations:
            self.annotations.save()

        if len(self.chip_file_paths) == 0:
            sys.exit(1)
        chip_file_path = self.chip_file_paths.pop()
        # This is the chip json file.  We need the annotation json file.
        self.chip_dir = os.path.split(chip_file_path)[0]
        with open(chip_file_path, 'r') as fp:
            meta = json.load(fp)
        scale = meta['scale']
        center = meta['center']
        error = meta['error']
        annotation_file_path = meta['json_file_path']
        # No need to read the chip. The viewer gets it from the sat image.
        #chip = os.path.join(chip_dir, os.path.split(meta['chip'])[1])
        #chip = cv2.imread(chip)

        annotations = Annotation(annotation_file_path)
        self.set_annotations(annotations)
        
        self.viewer.set_annotations(annotations)
        self.viewer.render(center, 0, scale)

        # TODO: Fix this.
        canvas_height, canvas_width = self.viewer.get_canvas_shape()

        # Update widgets that render info for the current plane.
        # I want to show the confidence for proofreading detections.
        score = float(error)
        score=str(round(score, 2))
        score_str = "score = %s"%score
        if self.viewer.text:
            self.viewer.canvas.itemconfig(self.viewer.text, text=score_str)
        else:
            self.viewer.text =  self.viewer.canvas.create_text(60,20,fill="darkblue",
                                                               font="Times 20 italic bold",
                                                               text=score_str)


    def set_annotations(self, annotations):
        """ 
        annotations: object with annotion elements already loaded.
        """
        self.viewer.set_annotations(annotations)
        self.annotations = annotations

        #elements = annotations.get_elements()
        #for plane in elements:


        






                
