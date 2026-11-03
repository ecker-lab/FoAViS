"""Run a pre-trained FoAViS model on a single image and look at its fixations.

    python -m foavis.predict photo.jpg --model imagenet_learned_s1 [--glimpses 4] [--out photo_foavis.png]

From Python (see demo.ipynb):

    from foavis.predict import load_model, predict, show
    model = load_model("ade20k_learned_s1")
    result = predict(model, "photo.jpg")
    show(result)

The checkpoint is downloaded on first use. ImageNet models classify the image (1000 classes), ADE20K and COCO-Stuff
models segment it; the C-MNIST models need C-MNIST images, see `cmnist_example`.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .data import class_names
from .data.loaders import IMAGENET_MEAN_STD
from .data.transforms import imagenet_transforms, segmentation_transforms
from .io.checkpoint import load_checkpoint
from .io.download import CHECKPOINTS, download_checkpoints
from .model.foavis import build_model
from .utils.parser import get_args_and_cfg

REPO = Path(__file__).resolve().parents[1]
# dataset -> (dataset_name in the configs, image channels, number of classes, image size)
DATASETS = {"cmnist": ("random_cluttered_MNIST", 1, 10, 224), "imagenet": ("imagenet", 3, 1000, 224),
            "ade20k": ("ade20k_semantic_seg", 3, 151, 512), "coco_stuff": ("coco_stuff", 3, 182, 512)}
CMNIST_MEAN_STD = ([0.1307], [0.3081])


@dataclass
class Prediction:
    image: np.ndarray          # (S, S, 3) or (S, S), the model input in [0, 1]
    fixations: np.ndarray      # (T, 2) the fixated locations in [-1, 1] image coordinates, (x, y); the first is the centre
    policy_logits: np.ndarray  # (T, G, G) policy-head logits after each glimpse (the last ones are not used);
                               # their softmax over all cells is the policy
    steps: np.ndarray          # classification: (T, classes) probabilities; segmentation: (T, S, S) label maps
    task: str                  # "classification" or "semantic_seg"
    policy: str                # where the fixations came from (see foavis/model/policy.py)
    class_names: list

    def top(self, k=5, step=-1):
        """Classification: the k most likely (class name, probability) after a step (default: the last)."""
        p = self.steps[step]
        return [(self.class_names[i], float(p[i])) for i in np.argsort(-p)[:k]]

    def classes_present(self, step=-1, min_fraction=0.01):
        """Segmentation: (class name, fraction of pixels) predicted after a step, largest first."""
        ids, counts = np.unique(self.steps[step], return_counts=True)
        frac = counts / counts.sum()
        return [(self.class_names[i], float(f)) for f, i in sorted(zip(frac, ids), reverse=True) if f >= min_fraction]


def _dataset_of(name):
    if name not in CHECKPOINTS:
        raise ValueError(f"unknown model {name!r}, one of {sorted(CHECKPOINTS)}")
    return name.rsplit("_", 2)[0]


def load_model(name, checkpoint_dir="checkpoints", num_glimpses=None, policy=None, device=None):
    """A released model, e.g. "imagenet_learned_s1", in evaluation mode. Downloads its checkpoint if missing.
    `num_glimpses` and `policy` override the trained setting (4 glimpses, its own policy)."""
    dataset, arm = _dataset_of(name), name.rsplit("_", 2)[1]
    config_name = f"imagenet_{arm}_phase2" if dataset == "imagenet" else f"{dataset}_{arm}"
    argv = ["--config_path", str(REPO / "configs" / "paper" / f"{config_name}.yaml"), "--just_evaluate", "true"]
    if num_glimpses is not None:
        argv += ["--num_glimpses", str(num_glimpses)]
    if policy is not None:
        argv += ["--policy", policy]
    config = get_args_and_cfg(str(REPO / "configs" / "config_default.yaml"), argv=argv)
    _, config.model.image_channels, config.model.num_classes, size = DATASETS[dataset]

    model = build_model(config, size)
    path = Path(checkpoint_dir) / f"{name}.pth"
    if not path.exists():
        download_checkpoints(str(checkpoint_dir), [name])
    load_checkpoint(model, str(path))
    model.dataset, model.image_size = dataset, size
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval()


def preprocess(image, dataset):
    """A PIL image or a path -> the normalized model input (1, C, S, S), as in the paper's evaluation."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    if dataset == "cmnist":
        raise ValueError("C-MNIST models take C-MNIST images, see cmnist_example()")
    image = image.convert("RGB")
    if dataset == "imagenet":
        tf = imagenet_transforms(224, IMAGENET_MEAN_STD, train=False)
    else:
        tf = segmentation_transforms(512, IMAGENET_MEAN_STD, train=False)
    return tf(image).unsqueeze(0)


def cmnist_example(index=0, data_dir="./data"):
    """(model input (1, 1, 224, 224), digit) of a C-MNIST validation image; MNIST and EMNIST are downloaded
    to data_dir on first use."""
    from .data.cmnist import ClutteredMNIST
    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*CMNIST_MEAN_STD)])
    sample = ClutteredMNIST(data_dir, False, norm, fixed_seed=True)[index]
    image, label = (sample["image"], sample["label"]) if isinstance(sample, dict) else sample[:2]
    return image.unsqueeze(0), int(label)


def _unnormalize(x, dataset):
    mean, std = CMNIST_MEAN_STD if dataset == "cmnist" else IMAGENET_MEAN_STD
    x = x[0].cpu() * torch.tensor(std).view(-1, 1, 1) + torch.tensor(mean).view(-1, 1, 1)
    x = x.clamp(0, 1).permute(1, 2, 0).numpy()
    return x[..., 0] if x.shape[-1] == 1 else x


@torch.no_grad()
def predict(model, image):
    """Run the model on one image (a path, a PIL image or a preprocessed (1, C, S, S) tensor)."""
    x = image if torch.is_tensor(image) else preprocess(image, model.dataset)
    device = next(model.parameters()).device
    out = model(x.to(device), training=False)
    T = model.num_glimpses
    task = "classification" if out.class_logits is not None else "semantic_seg"
    steps = out.class_logits[0].softmax(-1) if task == "classification" else out.seg_predictions[0]
    names = {"cmnist": class_names.CMNIST, "ade20k": class_names.ADE20K,
             "coco_stuff": class_names.COCO_STUFF}.get(model.dataset) or class_names.imagenet()
    return Prediction(image=_unnormalize(x, model.dataset), fixations=out.fixations[0, :T].cpu().numpy(),
                      policy_logits=out.policy_logits[0].cpu().numpy(),
                      steps=steps.cpu().numpy(), task=task, policy=model.policy, class_names=names)


def show(result, path=None):
    """Plot the image with the numbered fixations, the policy logits after each glimpse (except the last, which
    are not used) and the final prediction. Saves to `path` if given; returns the figure."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    T, S = len(result.fixations), result.image.shape[0]
    fig, axes = plt.subplots(1, T + 1, figsize=(2.4 * (T + 1), 3.1))
    xy = (result.fixations + 1) / 2 * (S - 1)
    cmap = "gray" if result.image.ndim == 2 else None
    ax = axes[0]
    ax.imshow(result.image, cmap=cmap)
    ax.plot(xy[:, 0], xy[:, 1], "-", color="white", lw=1)
    for t, (x, y) in enumerate(xy):
        ax.scatter(x, y, s=120, color="#cc79a7", edgecolor="white", zorder=3)
        ax.text(x, y, str(t + 1), color="white", ha="center", va="center", fontsize=7, zorder=4)
    ax.set_title("fixations", fontsize=9)
    for t in range(T - 1):
        axes[t + 1].imshow(result.policy_logits[t], cmap="magma", extent=(-0.5, S - 0.5, S - 0.5, -0.5))
        axes[t + 1].scatter(*xy[t + 1], marker="+", s=80, color="cyan")
        note = f"+ = fixation {t + 2}" if result.policy == "learned" else f"not used, + = {result.policy} fixation {t + 2}"
        axes[t + 1].set_title(f"policy logits, glimpse {t + 1}\n{note}", fontsize=9)
    ax = axes[T]
    if result.task == "classification":
        ax.imshow(result.image, cmap=cmap)
        ax.set_title("prediction", fontsize=9)
        ax.text(0.5, -0.04, "\n".join(f"{p:.0%} {n}" for n, p in result.top(3)), transform=ax.transAxes,
                ha="center", va="top", fontsize=8)
    else:
        present = [result.class_names.index(n) for n, _ in result.classes_present()][:8]
        colours = [c for i, c in enumerate(plt.get_cmap("tab10").colors) if i != 7]   # grey = other classes
        rgb = np.full(result.steps[-1].shape + (3,), 0.6)
        for i, c in enumerate(present):
            rgb[result.steps[-1] == c] = colours[i]
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title("segmentation", fontsize=9)
        ax.legend(handles=[Patch(color=colours[i], label=result.class_names[c]) for i, c in enumerate(present)],
                  loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=7, frameon=False)
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--model", default="imagenet_learned_s1", choices=sorted(CHECKPOINTS))
    ap.add_argument("--glimpses", type=int, default=None, help="default: 4, as trained")
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--out", default=None, help="figure path (default: <image>_foavis.png)")
    a = ap.parse_args()
    model = load_model(a.model, a.checkpoint_dir, num_glimpses=a.glimpses)
    result = predict(model, a.image)
    print("fixations (x, y in [-1, 1]):", ", ".join(f"({x:.2f}, {y:.2f})" for x, y in result.fixations))
    if result.task == "classification":
        for n, p in result.top(5):
            print(f"  {p:6.1%}  {n}")
    else:
        for n, f in result.classes_present():
            print(f"  {f:6.1%} of pixels  {n}")
    out = a.out or str(Path(a.image).with_suffix("")) + "_foavis.png"
    show(result, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
