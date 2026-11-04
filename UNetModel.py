from torch import nn
import torch.nn.functional as F
import torch

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(16)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample1 = nn.Upsample(32)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample2 = nn.Upsample(64)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample3 = nn.Upsample(128)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(128, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        skip0 = F.relu(self.enc_conv0(x))
        e0 = self.pool0(skip0)
        skip1 = F.relu(self.enc_conv1(e0))
        e1 = self.pool1(skip1)
        skip2 = F.relu(self.enc_conv2(e1))
        e2 = self.pool2(skip2)
        skip3 = F.relu(self.enc_conv3(e2))
        e3 = self.pool3(skip3)

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # decoder
        up0 = self.upsample0(b)
        cat0 = torch.concat([up0, skip3], dim=1)
        d0 = F.relu(self.dec_conv0(cat0))
        up1 = self.upsample1(d0)
        cat1 = torch.concat([up1, skip2], dim=1)
        d1 = F.relu(self.dec_conv1(cat1))
        up2 = self.upsample2(d1)
        cat2 = torch.concat([up2, skip1], dim=1)
        d2 = F.relu(self.dec_conv2(cat2))
        up3 = self.upsample3(d2)
        cat3 = torch.concat([up3, skip0], dim=1)
        d3 = self.dec_conv3(cat3)  # no activation
        return d3

class UNet2:
    pass