
from .networks import UNetGenerator, PatchDiscriminator
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from .utils import *

from tqdm import tqdm
import matplotlib.pyplot as plt


class DeCloudGAN:
    def __init__(self, opt: argparse.Namespace):
        self.opt = opt
        # TODO: worry about device selection later
        # self.device = opt.device

        self.G = UNetGenerator(
            input_nc=opt.input_nc, output_nc=opt.output_nc, n_encode=opt.n_encode, ngf=opt.ngf, num_downs=opt.num_downs,
            where_add=opt.g_where_add, norm=opt.norm, upsample=opt.upsample, use_dropout=opt.dropout)
        
        self.D = PatchDiscriminator(opt.output_nc, opt.n_encode, opt.ndf, opt.d_where_add, opt.norm)

        initialize_weights(self.G, self.opt.weights_init)
        initialize_weights(self.D, self.opt.weights_init)

        self.criterionGAN = nn.BCEWithLogitsLoss() if opt.gan_mode == 'vanilla' else nn.MSELoss()
        self.criterionL1 = nn.L1Loss()
        self.criterionSSIM = ssim_loss

        self.optimG = torch.optim.Adam(self.G.parameters(), opt.lr, (opt.beta1, opt.beta2))
        self.optimD = torch.optim.Adam(self.D.parameters(), opt.lr, (opt.beta1, opt.beta2))

    def train_D(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        self.optimD.zero_grad()

        if self.opt.gan_mode == 'wgangp':
            y_ = self.G(x, z)
            eps = torch.rand(y.size(0), 1, 1, 1).expand(-1, y.size(1), y.size(2), y.size(3))
            y_interp = eps * y + (1 - eps) * y_
            
            pred = self.D(y, z)
            pred_ = self.D(y_.detach(), z)

            pred_interp = self.D(y_interp, z)
            grad_outputs = torch.ones_like(pred_interp)
            grad = torch.autograd.grad(inputs=y_interp, outputs=pred_interp, grad_outputs=grad_outputs,
                                       create_graph=True, retain_graph=True)[0]
            penalty = ((grad.norm(2, dim=(1, 2, 3)) - 1) ** 2).mean()
        
            loss_d = pred_.mean() - pred.mean() + self.opt.lambda_gp * penalty

        elif self.opt.gan_mode in ('vanilla', 'lsgan'):
            y_ = self.G(x, z)
            pred = self.D(y, z)
            pred_ = self.D(y_.detach(), z)

            ones = torch.ones_like(pred)
            zeros = torch.zeros_like(pred)

            loss_d_real = self.criterionGAN(pred, zeros)
            loss_d_fake = self.criterionGAN(pred_, ones)
            loss_d = (loss_d_real + loss_d_fake) * 0.5 

        else:
            raise NotImplementedError

        loss_d.backward()
        self.optimD.step()

    def train_G(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        self.optimG.zero_grad()

        if self.opt.gan_mode == 'wgangp':
            y_ = self.G(x, z)
            pred_ = self.D(y_, z)

            loss_g = -pred_.mean()

        elif self.opt.gan_mode in ('vanilla', 'lsgan'):
            y_ = self.G(x, z)
            pred_ = self.D(y_, z)
            zeros = torch.zeros_like(pred_)

            loss_g_fake = self.criterionGAN(pred_, zeros)
            loss_l1 = self.criterionL1(y_, y)
            loss_ssim = self.criterionSSIM(y_, y)
            loss_g = loss_g_fake + self.opt.lambda_l1 * loss_l1 + self.opt.lambda_ssim * loss_ssim
            
        else:
            raise NotImplementedError
        
        loss_g.backward()
        self.optimG.step()

    @staticmethod
    def get_xyz(data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s1 = data['s1']
        s2 = data['s2']

        x = torch.cat([s1, s2[:,:-3]], dim=1)
        y = s2[:,-3:]

        s2_months = data['s2_months']
        target_month = s2_months[:,-1]
        angle = target_month * 12 / 2 * torch.pi
        z = torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)

        return x, y, torch.tensor([])


    def train_dataloader(self, dataloader: DataLoader):
        self.D.train()
        self.G.train()

        for i, data in enumerate(tqdm(dataloader)):
            x, y, z = self.get_xyz(data)

            self.train_D(x, y, z)

            if i % self.opt.n_critics == 0 or self.opt.gan_mode != 'wgangp':

                self.train_G(x, y, z)
    
    def valid_dataloader(self, dataloader: DataLoader):
        self.G.eval()

        l = 0
        n = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader)):
                x, y, z = self.get_xyz(data)
                y_ = self.G(x, z)

                l += -(ssim_loss(y, y_).item() - 1) * x.size(0)
                n += x.size(0)

                img = (y[0]+1)/2
                img = torch.movedim(img, 0, -1).numpy()

                plt.imsave(f'./results/{i}.png', img)

                img_ = (y_[0]+1)/2
                img_ = torch.movedim(img_, 0, -1).numpy()

                plt.imsave(f'./results/{i}_.png', img_)

        
        print(l / n)


                




 


