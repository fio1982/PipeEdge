# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn


class Stage2(nn.Module):
    def __init__(self):
        super(Stage2, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
        )
    

    def forward(self, x):
        x = self.classifier(x)
        return x   
