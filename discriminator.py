import torch
import torch.nn as nn
import torch.nn.functional as F


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

        time_embedding_dim = 32
        self.inc = (DoubleConv(n_channels, 64, time_embedding_dim))
        self.down1 = (Down(64, 128, time_embedding_dim))
        self.down2 = (Down(128, 256, time_embedding_dim))
        self.down3 = (Down(256, 512, time_embedding_dim))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, time_embedding_dim))
        self.up1 = (Up(1024, 512 // factor, time_embedding_dim, bilinear))
        self.up2 = (Up(512, 256 // factor, time_embedding_dim, bilinear))
        self.up3 = (Up(256, 128 // factor, time_embedding_dim, bilinear))
        self.up4 = (Up(128, 64, time_embedding_dim, bilinear))
        self.outc = (OutConv(64, n_classes))



        # time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x1 = self.inc(x, t)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)
        x5 = self.down4(x4, t)
        x = self.up1(x5, x4, t)
        x = self.up2(x, x3, t)
        x = self.up3(x, x2, t)
        x = self.up4(x, x1, t)
        logits = self.outc(x)
        return logits