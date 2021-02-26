"""
Copyright (c) 2019-present NAVER Corp.
"""

import torch.nn as nn
import torch.nn.init as init


# copied from basenet/vgg16_bn.py in the original repository
# not actually used (don't need to init weights because not training)
def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
