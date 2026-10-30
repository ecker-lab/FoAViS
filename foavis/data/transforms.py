"""Image transforms. Training: random-resized crops, flips (and colour jitter, grayscale, rotation for
segmentation). Evaluation: resize (and centre crop for ImageNet). Outputs are ImageNet-normalised."""
import torch
from torchvision.transforms import v2

BILINEAR = v2.InterpolationMode.BILINEAR


def _to_normalized_tensor(mean_std):
    return [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(*mean_std)]


def imagenet_transforms(size, mean_std, train):
    if train:
        geometry = [v2.RandomResizedCrop(size=size, scale=(0.08, 1.0), antialias=True), v2.RandomHorizontalFlip()]
    else:
        geometry = [v2.Resize(256, interpolation=BILINEAR), v2.CenterCrop(size)]
    return v2.Compose(geometry + _to_normalized_tensor(mean_std))


def segmentation_transforms(size, mean_std, train):
    """Applied jointly to the image and its label map (torchvision tv_tensors)."""
    if not train:
        return v2.Compose([v2.Resize((size, size), interpolation=BILINEAR, antialias=True),
                           *_to_normalized_tensor(mean_std)])
    return v2.Compose([
        v2.RandomResizedCrop(size=(size, size), scale=(0.4, 2.0), ratio=(0.75, 1.33), interpolation=BILINEAR,
                             antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomApply([v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
        v2.RandomGrayscale(p=0.1),
        v2.RandomRotation(degrees=10, interpolation=BILINEAR, fill=0),
        *_to_normalized_tensor(mean_std),
    ])
