
from typing import *

import torch
from torch import nn

from .utils import *

class UNetModule(nn.Module):
    def __init__(self, input_nc: int, output_nc: int, inner_nc: int, inner_module: Union['UNetModule', None],
                 encode_all: bool, upsample: str, is_outermost: bool=False, d_non_lin: str='lrelu', d_norm: str='instance',
                 d_dropout_rate: float=0, u_non_lin: str='relu', u_norm: str='instance', u_dropout_rate: float=0):
        super(UNetModule, self).__init__()

        d_bias = d_norm != 'batch' # BatchNorm2d has affine parameter
        d_non_lin = get_non_linearity(d_non_lin)
        d_norm = get_norm(d_norm)
        d_dropout = get_dropout(d_dropout_rate)

        u_bias = u_norm != 'batch'
        u_non_lin = get_non_linearity(u_non_lin)
        u_norm = get_norm(u_norm)
        u_dropout = get_dropout(u_dropout_rate)
        u_input_nc = inner_nc if inner_module is None else inner_nc * 2

        down = [
            nn.Conv2d(input_nc, inner_nc, 4, 2, 1, bias=d_bias, padding_mode='reflect'),
            d_norm(inner_nc),
            d_dropout(),
            d_non_lin()
        ]

        up = []
        if upsample == 'basic':
            up += [
                nn.ConvTranspose2d(u_input_nc, output_nc, 4, 2, 1, bias=u_bias)
                ]
        elif upsample == 'bilinear':
            up += [
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(u_input_nc, output_nc, 3, 1, 1, padding_mode='reflect')
            ]
        else:
            raise NotImplementedError
        
        up += [
            u_norm(output_nc),
            u_dropout(),
            u_non_lin()
        ]

        self.down = nn.Sequential(*down)
        self.inner_module = inner_module
        self.up = nn.Sequential(*up)

        self.is_outermost = is_outermost
        self.encode_all = encode_all

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # nc(y) = 2 * nc(x) if inner_module defined
        # nc(y) = nc(x) if inner_module not defined or is_outermost
        if z.numel() == 0:
            x_and_z = x
        else:
            z_encode = z.view(z.size(0), z.size(1), 1, 1).expand(-1, -1, x.size(2), x.size(3))
            x_and_z = torch.cat([x, z_encode], 1)

        y = self.down(x_and_z)
        if self.inner_module is not None: # not innermost
            y = self.inner_module(y, z if self.encode_all else torch.tensor([]))
        y = self.up(y)

        if self.is_outermost:
            return y
        return torch.cat([x, y], 1)


class UNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, n_encode, ngf=64, num_downs=8, where_add: str='all', norm='instance', upsample: str= 'basic', use_dropout=True):
        super(UNetGenerator, self).__init__()

        # enc C64-C128-C256-C512-C512-C512-C512-C512
        # dec CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
        if where_add == 'all':
            encode_all = True
        elif where_add == 'input':
            encode_all = False
        else:
            raise NotImplementedError

        z_nc = n_encode if encode_all else 0
        dropout_flag = int(use_dropout)

        innermost_norm = 'none' if norm == 'instance' else norm # InstanceNorm2d fails for 1x1 bottleneck
        unet_module = UNetModule(z_nc+ngf*8, ngf*8, ngf*8, None, encode_all, upsample, d_norm=innermost_norm, u_norm=norm, u_dropout_rate=dropout_flag*0.5)
        unet_module = UNetModule(z_nc+ngf*8, ngf*8, ngf*8, unet_module, encode_all, upsample, u_norm=norm, u_dropout_rate=dropout_flag*0.5)

        for i in range(num_downs-6):
            u_dropout_rate = 0.5 if i == 0 else 0
            unet_module = UNetModule(z_nc+ngf*8, ngf*8, ngf*8, unet_module, encode_all, upsample, u_norm=norm, u_dropout_rate=dropout_flag*u_dropout_rate)

        unet_module = UNetModule(z_nc+ngf*4, ngf*4, ngf*8, unet_module, encode_all, upsample, u_norm=norm)
        unet_module = UNetModule(z_nc+ngf*2, ngf*2, ngf*4, unet_module, encode_all, upsample, u_norm=norm)
        unet_module = UNetModule(z_nc+ngf, ngf, ngf*2, unet_module, encode_all, upsample, u_norm=norm)

        z_nc = n_encode

        unet_module = UNetModule(z_nc+input_nc, output_nc, ngf, unet_module, encode_all, upsample, is_outermost=True,
                                 d_norm='none', u_non_lin='tanh', u_norm=norm)
        self.unet_module = unet_module

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.unet_module(x, z)
    

class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc, n_encode, ndf=64, where_add: str='all', norm='instance'):
        super(PatchDiscriminator, self).__init__()
        if where_add == 'all':
            encode_all = True
        elif where_add == 'input':
            encode_all = False
        else:
            raise NotImplementedError

        z_nc = n_encode
        norm_layer = get_norm(norm)

        modules = []
        modules += [
            nn.Sequential(
                nn.Conv2d(z_nc+input_nc, ndf, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True)
                )
        ]

        z_nc = n_encode if encode_all else 0

        for i in range(3):
            j = 2 ** i
            stride = 2 if i in [0, 1] else 1
            modules += [
                nn.Sequential(
                    nn.Conv2d(z_nc+ndf*j, ndf*j*2, 4, stride, 1, bias=False),
                    norm_layer(ndf*j*2),
                    nn.LeakyReLU(0.2, inplace=True)
                    )
            ]
        
        modules += [nn.Conv2d(z_nc+ndf*8, 1, 4, 1, 1)]
        self.modules_list = nn.ModuleList(modules)

        self.encode_all = encode_all

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = x

        for module in self.modules_list:
            if z.numel() == 0:
                y_and_z = y
            else:
                z_encode = z.view(z.size(0), z.size(1), 1, 1).expand(-1, -1, y.size(2), y.size(3))
                y_and_z = torch.cat([y, z_encode], 1)

            y = module(y_and_z)

            if not self.encode_all:
                z = torch.tensor([])
        
        return y



if __name__ == '__main__':
    
    seed_everything(42)
    G = UNetGenerator(2, 3, 5, num_downs=8, upsample='bilinear', norm='instance')
    D = PatchDiscriminator(3, 5, where_add='input', norm='instance')

    #initialize_weights(G, 'normal')
    #initialize_weights(D, 'normal')

    x = torch.zeros((4, 2, 256, 256))
    z = torch.zeros((4, 5))

    MSE = nn.MSELoss()
    
    y = G(x, z)

    y_ = torch.rand((4, 3, 256, 256))

    pred_fake = D(y, z)
    pred_real = D(y_, z)

    ones = torch.ones_like(pred_fake)
    zeros = torch.zeros_like(pred_fake)

    lossD = MSE(pred_fake, ones) + MSE(pred_real, zeros)
    lossG = MSE(pred_fake, zeros)

    print(lossD)
    