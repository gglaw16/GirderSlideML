from __future__ import print_function
import pdb
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os


# duplicate fcnn116 but add skip paths over every group.
# skips are handled in forward and have no explicit layers.
# the only net change is that the number of components doubles
# in the last layer of each group.

class net(nn.Module):

    def __init__(self):
        super(net, self).__init__()

        layers = []

        self.components = 4

        self.conv_layer(layers, 32, 5)
        self.conv_layer(layers, 32, 5)
        self.conv_layer(layers, 32, 5)
        self.conv_layer(layers, 32, 5)
        layers.append(nn.MaxPool2d(2))
        self.components = 36 # (32 + 4)
        self.conv_layer(layers, 64, 5)
        self.conv_layer(layers, 64, 5)
        self.conv_layer(layers, 64, 5)
        self.conv_layer(layers, 64, 5)
        layers.append(nn.MaxPool2d(2))
        self.components = 100 # 64 + 32 + 4
        self.conv_layer(layers, 128, 5)
        self.conv_layer(layers, 128, 5)
        self.conv_layer(layers, 128, 5)
        self.conv_layer(layers, 128, 5)
        self.layers = nn.Sequential(*layers)

        layers = []
        self.components = 228 # # 128 + 64 + 32 + 4
        self.conv_layer(layers, 128, 1)
        layers.append(nn.Conv2d(128, 2, 1))
        self.end_layers = nn.Sequential(*layers)
        


    def conv_layer(self, layers, out_components, size):
        layers.append(nn.Conv2d(self.components, out_components, size))
        self.components = out_components

    def schedule_parameters(self):
        return self.parameters()
        
        
    def set_schedule(self, idx):
        print("no schedule")

        
    def get_rf_size(self):
        return 116

    
    def get_rf_stride(self):
        return 4

    
    def forward(self, x):

        skip = None
        for layer in self.layers:
            #print("{}     {}".format(type(layer), x.shape))
            if skip is None:
                skip = x
            if isinstance(layer, nn.MaxPool2d):
                margin = int((skip.shape[2] - x.shape[2]) / 2)
                skip = skip[:, :, margin:-margin, margin:-margin]
                x = torch.cat((x, skip), dim=1)
                skip = None
            x = layer(x)
            
        margin = int((skip.shape[2] - x.shape[2]) / 2)
        skip = skip[:, :, margin:-margin, margin:-margin]
        x = torch.cat((x, skip), dim=1)
        for layer in self.end_layers:
            #print("end: {}     {}".format(type(layer), x.shape))
            x = layer(x)
                
        return x
        

if __name__ == "__main__":
    net = net()

    for d in range(120,10, -1):
        print("input_size = %d"%d)
        in_img = Variable(torch.randn(2,3,d,d))
        output, _ = net(in_img)
        print(output.shape)

