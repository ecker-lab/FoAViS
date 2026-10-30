"""Download the released checkpoints from GRO.data (doi:10.25625/UXZTPA), one file DOI per checkpoint.

    python -m foavis.io.download                                  # all 8 into ./checkpoints
    python -m foavis.io.download --models cmnist_learned_s1       # a subset
"""
import argparse

from torchvision.datasets.utils import download_url

BASE_URL = "https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/UXZTPA/"

# name -> (file DOI suffix, MD5 as listed by GRO.data)
CHECKPOINTS = {
    "cmnist_random_s1": ("SAMCRS", "4aed93d83af7cdeccb20f9001eadb831"),
    "cmnist_learned_s1": ("TSME8T", "c89819a3004117410fb244e7aa429fe6"),
    "imagenet_random_s1": ("S1CQ8G", "f826c3c5358a1f01bfe92f100d151bb8"),
    "imagenet_learned_s1": ("PFLPPY", "4e246675c0674487c9205a7429203d13"),
    "ade20k_random_s1": ("V3X6ZM", "ea40acfa56182851342c59c6163cdb0f"),
    "ade20k_learned_s1": ("VZME3L", "5d4898db21803ef70c108dddf1531182"),
    "coco_stuff_random_s1": ("FI81IA", "a242ad1dac947e0f3a66f74aa05462aa"),
    "coco_stuff_learned_s1": ("BBFUZJ", "a8f05608b62a4d46221237fcaba381f9"),
}


def download_checkpoints(root="checkpoints", models=None):
    """Download <root>/<name>.pth for each name in `models` (default: all). A file already present with the
    right MD5 is kept; the MD5 is checked after every download."""
    for name in models or CHECKPOINTS:
        doi, md5 = CHECKPOINTS[name]
        download_url(BASE_URL + doi, root, name + ".pth", md5=md5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="checkpoints")
    ap.add_argument("--models", nargs="+", choices=list(CHECKPOINTS), default=None)
    a = ap.parse_args()
    download_checkpoints(a.root, a.models)
