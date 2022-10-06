# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn

class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    

    def forward(self, x):       
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        return x
