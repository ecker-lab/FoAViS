"""ImageNet (ILSVRC-2012) classification: <root>/ImageNet/{train_images,val}/<class folder>/<image>."""
import glob
import os

from PIL import Image
from torch.utils.data import Dataset


class ImageNet(Dataset):
    def __init__(self, root, split, transform):
        folder = os.path.join(root, "ImageNet", "train_images" if split == "train" else "val")
        classes = sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))
        self.samples = [(p, label) for label, c in enumerate(classes) for p in glob.glob(os.path.join(folder, c, "*.*"))
                        if os.path.splitext(p)[-1].lower() in (".jpg", ".jpeg", ".png")]
        if not self.samples:
            raise FileNotFoundError(f"no images in {folder}")
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return {"image": self.transform(Image.open(path).convert("RGB")), "label": label, "image_path": path}
