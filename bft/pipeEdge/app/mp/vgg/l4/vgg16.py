"Modified from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py"

import torch
import torch.nn as nn
from .stage0 import Stage0
from .stage1 import Stage1

class VGGSplit(nn.Module):

    def __init__(self) -> None:
        super(VGGSplit, self).__init__()
        self.stage0 = Stage0()
        # self.stage1 = Stage1()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out0 = self.stage0(x)
        return out0
        # out1 = self.stage1(out0)
        # return out1 

def vgg16l4_1():
    return VGGSplit()



