"""Policy head (paper Sec. 3.4) and task heads (Sec. 3.5), all reading the memory features H_t."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHead(nn.Module):
    """psi_theta: `layers` 3x3 convolutions with `channels` channels and ReLU, then a 1x1 convolution to one
    channel. Returns the policy logits (B, G, G); softmax over the grid gives pi(. | H_t).
    channels=0 gives a single 1x1 convolution (the small-head ablation)."""

    def __init__(self, input_channels, channels=128, layers=3):
        super().__init__()
        if channels:
            net, c_in = [], input_channels
            for _ in range(max(1, int(layers))):
                net += [nn.Conv2d(c_in, channels, kernel_size=3, padding=1, bias=True), nn.ReLU(inplace=True)]
                c_in = channels
            self.net = nn.Sequential(*net, nn.Conv2d(channels, 1, kernel_size=1, bias=True))
        else:
            self.net = nn.Conv2d(input_channels, 1, kernel_size=1, bias=True)

    def forward(self, memory):
        return self.net(memory).squeeze(1)


class ClassificationHead(nn.Module):
    """phi_theta for classification: two 3x3 convolutions (2C channels, the first with stride 2), global
    average pooling, and a two-layer MLP."""

    def __init__(self, input_channels, num_classes):
        super().__init__()
        c = input_channels * 2
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, c, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1))
        self.mlp = nn.Sequential(nn.Flatten(), nn.Linear(c, 256), nn.ReLU(inplace=True), nn.Linear(256, num_classes))

    def forward(self, memory):
        return self.mlp(self.conv(memory))


class SegmentationHead(nn.Module):
    """phi_theta for segmentation: a 3x3 convolution with ReLU and a 1x1 convolution to the classes, then an
    upsampling of the logits to the image resolution."""

    def __init__(self, input_channels, num_classes, output_size):
        super().__init__()
        self.output_size = output_size
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, num_classes, kernel_size=1))

    def forward(self, memory, mode='bilinear'):
        return self.upsample(self.conv(memory), mode)

    def upsample(self, logits, mode='bilinear'):
        # always fp32: the backward of a bf16 bilinear upsample is several times slower
        with torch.autocast(device_type=logits.device.type, enabled=False):
            kw = {'align_corners': False} if mode == 'bilinear' else {}
            return F.interpolate(logits.float(), size=(self.output_size, self.output_size), mode=mode, **kw)
