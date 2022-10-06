"""google net in pytorch



[1] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.

    Going Deeper with Convolutions
    https://arxiv.org/abs/1409.4842v1
"""

import torch
import torch.nn as nn
from .stage0 import Stage0
from .stage1 import Stage1

class AlexNetSplit(nn.Module):
    def __init__(self):
        super(AlexNetSplit, self).__init__()
        self.stage0 = Stage0()
        # self.stage1 = Stage1()

    def forward(self, x):
        out0 = self.stage0(x)
        return out0
        # out1 = self.stage1(out0)
        # return out1  

def alexnetl2_1():
    return AlexNetSplit()


