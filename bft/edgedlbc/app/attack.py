import copy
import torch
from torch import nn
from models import CNNMnist
import numpy as np


def add_gaussian_noise(w, device, scale): #scale默认100
    #device=torch.device("cuda:1")
    w_attacked = copy.deepcopy(w)
    if type(w_attacked) == list:
        for k in range(len(w_attacked)):
            noise = torch.randn(w[k].shape).cuda() * scale / 100.0 * w_attacked[k]
            w_attacked[k] += noise
    else:
        for k in w_attacked.keys():
            noise = torch.randn(w[k].shape).to(device) * scale / 100.0 * w_attacked[k].to(device)
            #noise = torch.randn(w[k].shape) * scale / 100.0 * w_attacked[k]
            w_attacked[k] = w_attacked[k].to(device).float() + noise
    return w_attacked

def sign_flipping_attack(w, p):   # p都是负数，-1，-2，-3，-4. 默认-3
    w_attacked = copy.deepcopy(w)
    if type(w_attacked) == list:
        for k in range(len(w_attacked)):
            w_attacked[k] *= p
    else:
        for k in w_attacked.keys():
            w_attacked[k] *= p

    return w_attacked

# if __name__ == '__main__':
#     model = CNNMnist().state_dict()
#     gus_model = sign_flipping_attack(model, p=-1)
#     print(gus_model)

