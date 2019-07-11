from __future__ import print_function
import pdb
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os
import math

class net(nn.Module):

    def __init__(self):
        super(net, self).__init__()

        layers = []

        self.network_design = [0,-1,1,-1,2,2,-1,3,3]
        self.components = 3
        self.shape = 7
        self.num_start_channels = 64

        for l in self.network_design:
            if l == -1:
                layers.append(nn.MaxPool2d(2))
            else:
                self.conv_layer(layers, self.num_start_channels*math.pow(2,l), self.shape)
               
        self.conv_layer(layers, 2, 1)
        
        self.layers = nn.Sequential(*layers)


    def conv_layer(self, layers, out_components, size):
        layers.append(nn.Conv2d(self.components, out_components, size))
        self.components = out_components
        layers.append(nn.BatchNorm2d(out_components, affine=False))
        layers.append(nn.LeakyReLU())


        
        
    def set_schedule(self, idx):
        print("no schedule")

        
    def get_rf_size(self):
        return 116

    
    def get_rf_stride(self):
        return 4

    
    def forward(self, x):

        for layer in self.layers:
            if True: #not isinstance(layer, nn.BatchNorm2d):
                x = layer(x)
            
        return x
        

if __name__ == "__main__":
    net = net()

    for d in range(188, 0,-1):
        print("input_size = %d"%d)
        in_img = Variable(torch.randn(2,3,d,d))
        output, _ = net(in_img)
        print(output.shape)

