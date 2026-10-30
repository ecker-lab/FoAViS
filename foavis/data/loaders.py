"""Datasets and data loaders of the paper. Sets the dataset-dependent model settings (model.image_channels,
model.num_classes, model.seg_ignore_index) in the config."""

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from .cmnist import ClutteredMNIST
from .imagenet import ImageNet
from .segmentation import SegmentationDataset
from .transforms import imagenet_transforms, segmentation_transforms
from ..utils.distributed import worker_init_fn

IMAGENET_MEAN_STD = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]


def _datasets(config):
    """(train, val, image size); sets the dataset-dependent model settings in the config."""
    name, root, m = config.training.dataset_name, config.dataset.data_dir, config.model
    if name == "random_cluttered_MNIST":
        m.image_channels, m.num_classes = 1, 10
        norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])])
        return ClutteredMNIST(root, True, norm), ClutteredMNIST(root, False, norm, fixed_seed=True), 224
    m.image_channels = 3
    if name == "imagenet":
        m.num_classes = 1000
        return (ImageNet(root, "train", imagenet_transforms(224, IMAGENET_MEAN_STD, train=True)),
                ImageNet(root, "val", imagenet_transforms(224, IMAGENET_MEAN_STD, train=False)), 224)
    train = SegmentationDataset(name, root, "train", segmentation_transforms(512, IMAGENET_MEAN_STD, True))
    val = SegmentationDataset(name, root, "val", segmentation_transforms(512, IMAGENET_MEAN_STD, False))
    m.num_classes, m.seg_ignore_index = train.num_classes, train.ignore_index
    return train, val, 512


def setup_dataloader(config, dist_info=None):
    """Returns train_loader, valid_loader, image_size."""
    train, val, size = _datasets(config)

    distributed = dist_info is not None and dist_info['is_distributed']
    samplers = [DistributedSampler(d, num_replicas=dist_info['world_size'], rank=dist_info['rank'], shuffle=True,
                                   drop_last=True) if distributed else None for d in (train, val)]
    rank = dist_info['rank'] if dist_info is not None else 0
    workers = config.dataloader.num_workers
    kwargs = dict(batch_size=config.dataloader.batch_size, pin_memory=True, num_workers=workers,
                  worker_init_fn=worker_init_fn((config.utils.seed or 0) + rank),
                  **({"persistent_workers": True, "prefetch_factor": 2} if workers > 0 else {}))
    valid_loader = DataLoader(val, shuffle=not distributed, sampler=samplers[1], drop_last=False, **kwargs)
    train_loader = DataLoader(train, shuffle=not distributed, sampler=samplers[0], drop_last=True, **kwargs)
    return train_loader, valid_loader, size
