
import functools
from typing import *

import torch
from torch import nn
from torch.nn import init

from pytorch_msssim import ssim

def ssim_loss(x: torch.Tensor, y: torch.Tensor):
    # x and y must be rescaled between 0 and 1.
    return 1 - ssim(x, y, data_range=1, win_size=11, win_sigma=1.5, nonnegative_ssim=True)

def get_non_linearity(name: str) -> nn.Module:
    if name == 'lrelu':
        return functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif name == 'relu':
        return functools.partial(nn.ReLU, inplace=True)
    elif name == 'tanh':
        return nn.Tanh
    elif name == 'sig':
        return nn.Sigmoid
    elif name == 'none':
        return nn.Identity
    else:
        raise NotImplementedError

def get_norm(name: str) -> nn.Module:
    if name == 'batch':
        return nn.BatchNorm2d
    elif name == 'instance':
        return nn.InstanceNorm2d
    elif name == 'none':
        return nn.Identity
    else:
        raise NotImplementedError
    
def get_dropout(dropout_rate: float) -> nn.Module:
    if dropout_rate == 0.:
        return nn.Identity
    else:
        return functools.partial(nn.Dropout, p=dropout_rate, inplace=True)

# https://github.com/junyanz/BicycleGAN/blob/master/models/networks.py
def initialize_weights(net: nn.Module, initialization: str='normal'):
    def init_func(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) and hasattr(m, 'weight'):
            if initialization == 'normal':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif initialization == 'xavier':
                init.xavier_normal_(m.weight.data, gain=0.02)
            elif initialization == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif initialization == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=0.02)
            else:
                raise NotImplementedError
            
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

# https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
def seed_everything(seed: int):
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True