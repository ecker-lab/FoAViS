"""Passive baseline (paper Sec. 4, Appendix "Passive baseline"): the Res-UNet backbone and the task head of
FoAViS on the whole image in a single forward pass, without foveation, memory or policy.

The image is resized to image_size x image_size (128: the pixels of four 64 x 64 glimpses); the task head
reads the backbone output at that resolution. Returns the same Outputs as FoAViS with one step (T = 1),
so the trainer and the losses are shared.
"""
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ResUNetBackbone
from .foavis import Outputs
from .heads import ClassificationHead, SegmentationHead


class PassiveBaseline(nn.Module):
    def __init__(self, task, num_classes, image_size, input_size=128, backbone=None, image_channels=3):
        super().__init__()
        self.task, self.input_size = task, input_size
        self.backbone = ResUNetBackbone(image_channels, **(backbone or {}))
        self.classification_head = self.segmentation_head = None
        if task == "classification":
            self.classification_head = ClassificationHead(self.backbone.out_channels, num_classes)
        else:
            self.segmentation_head = SegmentationHead(self.backbone.out_channels, num_classes, image_size)

    def forward(self, images, training, fixations=None):
        x = F.interpolate(images, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False,
                          antialias=True)
        features = self.backbone(x)
        if self.classification_head is not None:
            return Outputs(class_logits=self.classification_head(features).unsqueeze(1))
        seg = self.segmentation_head(features).unsqueeze(1)                    # (B, 1, classes, S, S)
        return Outputs(seg_logits=seg, seg_predictions=None if training else seg[:, 0].argmax(1).unsqueeze(1))
