# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:57:27 2021
@author: xiaoxiaoyang
"""

import torch.nn as nn
from torchvision import models

class BuildAlexNet(nn.Module):
    def __init__(self, model_type, n_output):
        super(BuildAlexNet, self).__init__()
        self.model_type = model_type
        if model_type == 'pre':
            model = models.alexnet(pretrained=True)
            self.features = model.features
            fc1 = nn.Linear(9216, 4096)
            fc1.bias = model.classifier[1].bias
            fc1.weight = model.classifier[1].weight
            
            fc2 = nn.Linear(4096, 4096)
            fc2.bias = model.classifier[4].bias
            fc2.weight = model.classifier[4].weight
            
            self.classifier = nn.Sequential(
                    nn.Dropout(),
                    fc1,
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    fc2,
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, n_output))  
            #或者直接修改为
#            model.classifier[6]==nn.Linear(4096,n_output)
#            self.classifier = model.classifier
        if model_type == 'new':
            self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 11, 4, 2),
                    nn.ReLU(inplace = True),
                    nn.MaxPool2d(3, 2, 0),
                    nn.Conv2d(64, 192, 5, 1, 2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, 2, 0),
                    nn.Conv2d(192, 384, 3, 1, 1),
                    nn.ReLU(inplace = True),
                    nn.Conv2d(384, 256, 3, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, 2, 0))
            self.classifier = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(9216, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, n_output))
            
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out  = self.classifier(x)
        return out

import numpy as np
from torch.autograd import Variable
import torch

if __name__ == '__main__':
    model_type = 'pre'
    n_output = 7
    alexnet = BuildAlexNet(model_type, n_output)
    print(alexnet)
    
    x = np.random.rand(1,3,224,224)
    x = x.astype(np.float32)
    x_ts = torch.from_numpy(x)
    x_in = Variable(x_ts)
    y = alexnet(x_in)
    
    print(y)



