import torch
import torch.nn as nn
import torch.nn.functional as F
from unet_parts import *
import math
import numpy as np

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim//2
        embeddings = math.log(10000)/ (half_dim -1)
        embeddings = torch.exp(torch.arange(half_dim, device=device)* -embeddings)
        embeddings = time[:,None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, apply_norm=True):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 4, stride, padding=1)
        self.norm = nn.InstanceNorm2d(out_ch)
        self.relu = nn.LeakyReLU(0.2, True)
        self.apply_norm = apply_norm

    def forward(self, x):
        x = self.conv(x)
        if self.apply_norm:
            x = self.norm(x)
        x = self.relu(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, img_ch):
        super(Discriminator, self).__init__()
        self.img_ch = img_ch

        # splitting images into 70x70 patches is done implicitly, calculate receptive field in reverse order to confirm

        self.conv1 = Block(img_ch, 64, apply_norm=False) # 128
        self.conv2 = Block(64, 128) # 64
        self.conv3 = Block(128, 256) # 32

        self.conv4 = Block(256, 512, stride=1) # 30
        self.out = nn.Conv2d(512, 1, 4, padding=1) # 30

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.out(x)

        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleBlock(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (DoubleBlock(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up2(128, 64))
        self.outc = (OutBlock(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits