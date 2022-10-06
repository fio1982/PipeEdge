# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn

class Stage1(nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.layer35 = torch.nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.layer36 = torch.nn.ReLU(inplace=True)
        self.layer37 = torch.nn.Dropout(p=0.5)
        self.layer38 = torch.nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.layer39 = torch.nn.ReLU(inplace=True)
        self.layer40 = torch.nn.Dropout(p=0.5)
        self.layer41 = torch.nn.Linear(in_features=4096, out_features=1000, bias=True)

        self._initialize_weights()
    

    def forward(self, x):
        out35 = self.layer35(x)
        out36 = self.layer36(out35)
        out37 = self.layer37(out36)
        out38 = self.layer38(out37)
        out39 = self.layer39(out38)
        out40 = self.layer40(out39)
        out41 = self.layer41(out40)
        return out41  

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)