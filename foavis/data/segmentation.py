"""Semantic segmentation datasets: images (.jpg) with single-channel label PNGs of the same name.

ADE20K (SceneParsing):  <root>/ADEChallengeData2016/{images,annotations}/{training,validation}/
                        151 labels, 0 = void (ignored), 1..150 the classes.
COCO-Stuff 164k:        images <root>/coco2017/{train2017,val2017}/, labels (stuffthingmaps)
                        <root>/coco_stuff/{train2017,val2017}/; 182 labels 0..181, 255 = unlabeled (ignored).
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.tv_tensors import Image as TVImage, Mask

LABELS = {"ade20k_semantic_seg": (151, 0), "coco_stuff": (182, 255)}   # (number of labels, ignored label)


def directories(name, split):
    """Image and label directories relative to data_dir."""
    if name == "ade20k_semantic_seg":
        s = "training" if split == "train" else "validation"
        return f"ADEChallengeData2016/images/{s}", f"ADEChallengeData2016/annotations/{s}"
    return f"coco2017/{split}2017", f"coco_stuff/{split}2017"


class SegmentationDataset(Dataset):
    def __init__(self, name, root, split, transform):
        self.num_classes, self.ignore_index = LABELS[name]
        images, labels = directories(name, split)
        self.image_dir, self.label_dir = Path(root, images), Path(root, labels)
        self.image_paths = sorted(self.image_dir.glob("*.jpg"))
        if not self.image_paths:
            raise FileNotFoundError(f"no .jpg images in {self.image_dir}")
        self.transform = transform   # applied jointly to image and label map

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = TVImage(np.array(Image.open(path).convert("RGB")).transpose(2, 0, 1))
        label = Mask(torch.from_numpy(np.array(Image.open(self.label_dir / (path.stem + ".png"))).astype(np.int64)))
        image, label = self.transform(image, label)
        return {"image": image, "label_map": label.squeeze(0).long(), "image_path": str(path)}
