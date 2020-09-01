#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetworkModel(nn.Module):  # Net
    def __init__(self, image_size, kernel=5):
        super().__init__()
        self.image_size = image_size
        self.kernel = kernel

        self.conv11 = nn.Conv2d(3, 6, 5, 1)
        self.conv12 = nn.Conv2d(6, 6, 5, 1)
        self.conv13 = nn.Conv2d(6, 6, 5, 1)
        self.conv14 = nn.Conv2d(6, 6, 5, 1)
        self.conv15 = nn.Conv2d(6, 6, 5, 1)
        self.fc11 = nn.Linear(150, 10)

        self.conv21 = nn.Conv2d(3, 6, 5, 1)
        self.conv22 = nn.Conv2d(6, 6, 5, 1)
        self.conv23 = nn.Conv2d(6, 6, 5, 1)
        self.conv24 = nn.Conv2d(6, 6, 5, 1)
        self.conv25 = nn.Conv2d(6, 6, 5, 1)
        self.fc21 = nn.Linear(150, 10)

        self.fc_out = nn.Linear(20, 2)

        #self.conv1 = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        #self.fc1 = nn.Linear(image_size, image_size)
        #self.fc2 = nn.Linear(image_size, image_size)
        #self.fc3 = nn.Linear(image_size, 1)
        #self.conv1.weight = nn.Parameter(kernel)

    def forward(self, x):
        ##x = x.view(-1, 7*7*3)
        ##x = F.tanhshrink(self.fc1(x))
        ##x = F.relu(self.fc2(x))
        ##x = F.sigmoid(self.fc3(x))
        ##x = self.fc3(x)

        #print(x.shape)
        x1 = x[:,:,:self.image_size,:]
        x2 = x[:,:,self.image_size:,:] #image_size

        x1 = F.relu(self.conv11(x1)) # [1, 6, 96, 96]
        x1 = F.max_pool2d(x1, 2, 2)  # [1, 6, 48, 48]
        x1 = F.relu(self.conv12(x1)) # [1, 6, 44, 44]
        x1 = F.max_pool2d(x1, 2, 2)  # [1, 6, 22, 22]
        x1 = F.relu(self.conv13(x1)) # [1, 6, 18, 18]
        x1 = F.max_pool2d(x1, 2, 2)  # [1, 6, 9, 9])
        x1 = F.relu(self.conv14(x1)) # [1, 6, 5, 5]
        x1 = x1.view(-1, 150)
        x1 = self.fc11(x1)

        x2 = F.relu(self.conv21(x2))
        x2 = F.max_pool2d(x2, 2, 2)
        x2 = F.relu(self.conv22(x2))
        x2 = F.max_pool2d(x2, 2, 2)
        x2 = F.relu(self.conv23(x2))
        x2 = F.max_pool2d(x2, 2, 2)
        x2 = F.relu(self.conv24(x2))
        x2 = x2.view(-1, 150)
        x2 = self.fc21(x2)

        x = torch.cat((x1, x2), 1)  # [1, 20]
        x = self.fc_out(x)
        x = F.log_softmax(x, dim=1)
        return x
