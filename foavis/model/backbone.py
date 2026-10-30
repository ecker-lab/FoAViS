"""Backbone (paper Sec. 3.2): Res-UNet encoder-decoder on the 64 x 64 log-polar glimpse."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with instance norm; a 1x1 projection shortcut when shape changes."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_ch, affine=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_ch, affine=True)
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                                          nn.InstanceNorm2d(out_ch, affine=True))

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)), inplace=True)
        out = self.norm2(self.conv2(out))
        out += x if self.shortcut is None else self.shortcut(x)
        return F.relu(out, inplace=True)


class FuseBlock(nn.Module):
    """Decoder stage: 1x1 fusion of [upsampled features, encoder skip] to the decoder width, then a ResidualBlock."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.fuse = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.fuse_norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.block = ResidualBlock(out_ch, out_ch)

    def forward(self, x):
        return self.block(F.relu(self.fuse_norm(self.fuse(x)), inplace=True))


class ResUNetBackbone(nn.Module):
    """Encoder: four stages of `blocks_per_stage` ResidualBlocks with C, 2C, 4C, 8C channels; stages 2-4 halve
    the resolution. Bottleneck: a ResidualBlock 8C -> D at the resolution of the last stage. Decoder: four
    FuseBlocks at constant width D, back to the input resolution. Returns (B, D, 64, 64) features.
    """

    def __init__(self, input_channels, channels=64, blocks_per_stage=2, decoder_channels=128):
        super().__init__()
        C, D = channels, decoder_channels
        enc = (C, 2 * C, 4 * C, 8 * C)

        def stage(cin, cout, stride):
            return nn.Sequential(ResidualBlock(cin, cout, stride),
                                 *[ResidualBlock(cout, cout) for _ in range(1, blocks_per_stage)])

        self.enc1 = stage(input_channels, enc[0], 1)
        self.enc2 = stage(enc[0], enc[1], 2)
        self.enc3 = stage(enc[1], enc[2], 2)
        self.enc4 = stage(enc[2], enc[3], 2)
        self.bottleneck = ResidualBlock(enc[3], D)
        self.dec4 = FuseBlock(D + enc[3], D)
        self.upconv3 = nn.ConvTranspose2d(D, D, kernel_size=2, stride=2)
        self.dec3 = FuseBlock(D + enc[2], D)
        self.upconv2 = nn.ConvTranspose2d(D, D, kernel_size=2, stride=2)
        self.dec2 = FuseBlock(D + enc[1], D)
        self.upconv1 = nn.ConvTranspose2d(D, D, kernel_size=2, stride=2)
        self.dec1 = FuseBlock(D + enc[0], D)
        self.out_channels = D

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d4 = self.dec4(torch.cat([self.bottleneck(e4), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        return self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))
