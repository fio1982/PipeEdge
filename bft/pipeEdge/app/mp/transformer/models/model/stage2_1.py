# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
from .decoder import Decoder

class Stage1(nn.Module):
    def __init__(self, d_model, n_head, max_len, ffn_hidden, dec_voc_size, drop_prob, n_layers, device):
        super(Stage1, self).__init__()
        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)
        

    def forward(self, trg, enc_src, trg_mask, src_trg_mask):
        output = self.decoder(trg, enc_src, trg_mask, src_trg_mask)
        return output 
