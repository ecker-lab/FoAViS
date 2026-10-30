"""C-MNIST: one MNIST digit (the target, 10 classes) and EMNIST letters (clutter) at random positions on a
black 224 x 224 canvas. Generated on the fly from the torchvision datasets (downloaded to `root`).

Training images are new random composites every epoch. Validation images (fixed_seed) are a pure function of
the index, so every evaluation sees the same set and recorded scanpaths can be replayed (key: str(idx)).
"""
import random

import numpy as np
import torch
import torchvision.datasets as tvds
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class ClutteredMNIST(Dataset):
    def __init__(self, root, train, transform, fixed_seed=False, canvas_size=224, num_letters=5):
        self.digits = tvds.MNIST(root, train=train, download=True, transform=T.ToTensor())
        self.letters = tvds.EMNIST(root, split="letters", train=train, download=True, transform=T.ToTensor())
        self.transform, self.fixed_seed = transform, fixed_seed
        self.canvas_size, self.num_letters = canvas_size, num_letters

    def __len__(self):
        return len(self.digits)

    def __getitem__(self, idx):
        rng = random.Random(idx) if self.fixed_seed else random
        digit, label = self.digits[idx]
        letters = [self.letters[rng.randint(0, len(self.letters) - 1)][0] for _ in range(self.num_letters)]
        canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        for obj in [digit, *letters]:   # the target first, overlaps keep the brighter pixel
            arr = (obj[0].numpy() * 255).astype(np.uint8)
            h, w = arr.shape
            x, y = rng.randint(0, self.canvas_size - w), rng.randint(0, self.canvas_size - h)
            canvas[y:y + h, x:x + w] = np.maximum(canvas[y:y + h, x:x + w], arr)
        sample = {"image": self.transform(Image.fromarray(canvas, mode="L")), "label": torch.tensor(label)}
        if self.fixed_seed:
            sample["image_path"] = str(idx)
        return sample
