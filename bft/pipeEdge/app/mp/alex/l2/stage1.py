# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn

NUM_CLASSES = 10
class Stage1(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(Stage1, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    

    def forward(self, x):
        x = self.classifier(x)
        return x   
