# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn

class Stage0(torch.nn.Module):
    def __init__(self, features):
        super(Stage0, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):       
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
