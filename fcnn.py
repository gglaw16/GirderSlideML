from __future__ import print_function
import pdb
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os

class net(nn.Module):

    def __init__(self):
        super(net, self).__init__()

        self.schedule = []
        rf_size = 1
        rf_stride = 1
        
        layers = []

    # TODO: add a fourth component? lower resolution results
        components = 4
        out_components = 32
        for i in range(4):
            layers.append(nn.Conv2d(components, out_components, 5))
            layers.append(nn.BatchNorm2d(out_components, affine=False))
            layers.append(nn.LeakyReLU())
            components = out_components
            rf_size += 4*rf_stride
            self.schedule.append({'stride':rf_stride, 'size':rf_size, 'layers':len(layers)})
        # (18)

        layers.append(nn.MaxPool2d(2))
        rf_size += rf_stride
        rf_stride = rf_stride * 2
        
        out_components = 64
        for i in range(4):
            layers.append(nn.Conv2d(components, out_components, 5))
            layers.append(nn.BatchNorm2d(out_components, affine=False))
            layers.append(nn.LeakyReLU())
            components = out_components
            rf_size += 4*rf_stride
            self.schedule.append({'stride':rf_stride, 'size':rf_size, 'layers':len(layers)})
        # (31)

        layers.append(nn.MaxPool2d(2))
        rf_size += rf_stride
        rf_stride = rf_stride * 2

        out_components = 128
        for i in range(4):
            layers.append(nn.Conv2d(components, out_components, 5))
            layers.append(nn.BatchNorm2d(out_components, affine=False))
            layers.append(nn.LeakyReLU())
            components = out_components
            rf_size += 4*rf_stride
            self.schedule.append({'stride':rf_stride, 'size':rf_size, 'layers':len(layers)})            

        self.num_pre_layers = len(layers)

        # Layers that are on the end of all scheduled layers.
        layers.append(nn.Conv2d(components, components, 1))
        layers.append(nn.BatchNorm2d(components, affine=False))
        layers.append(nn.LeakyReLU())
        # final layer to get the output parameters
        # no relu or softmax (of course).
        # I put an extra NIN, because this layer cannot do much logic.
        layers.append(nn.Conv2d(components, 2, 1))
        
        self.layers = nn.Sequential(*layers)

        # default to executing all layers
        self.schedule_idx = len(self.schedule) - 1 

        
    def set_schedule(self, idx):
        if idx >= len(self.schedule):
            idx = len(self.schedule) - 1
        self.schedule_idx = idx

        
    def get_rf_size(self):
        schedule = self.schedule[self.schedule_idx]
        return schedule['size']

    
    def get_rf_stride(self):
        schedule = self.schedule[self.schedule_idx]
        return schedule['stride']

    
    def forward(self, x):
        schedule = self.schedule[self.schedule_idx]
        num_layers = schedule['layers']

        for i in range(num_layers):
            layer = self.layers[i]
            if not isinstance(layer, nn.BatchNorm2d):
                x = layer(x)

        batch_size, num_comps, dy, dx = x.shape
        if num_comps != 128:
            pad = Variable(torch.zeros(batch_size, 128-num_comps, dy, dx))
            if torch.cuda.is_available():
                pad = pad.cuda(0)
            x = torch.cat((x, pad), 1)   
            
        # Post layers that get executed for every schedule.
        for i in range(self.num_pre_layers, len(self.layers)):
            layer = self.layers[i]
            if not isinstance(layer, nn.BatchNorm2d):
                x = layer(x)    
        
        return x
        

if __name__ == "__main__":
    net = net()
    for idx in range(len(net.schedule)):
        net.set_schedule(idx)
        rf_size = net.get_rf_size() 
        print("%d: %d"%(idx, rf_size))
        

    d = 116
    print("input_size = %d"%d)
    in_img = Variable(torch.randn(2,3,d,d))
    output, _ = net(in_img)
    print(output.shape)
    #print(net)

