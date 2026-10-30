import os

import pytest

from foavis.io.download import CHECKPOINTS, download_checkpoints


def test_one_entry_per_model():
    expected = {f"{d}_{a}_s1" for d in ("cmnist", "imagenet", "ade20k", "coco_stuff") for a in ("random", "learned")}
    assert set(CHECKPOINTS) == expected
    assert len({doi for doi, _ in CHECKPOINTS.values()}) == len(CHECKPOINTS)


@pytest.mark.skipif(not os.environ.get("FOAVIS_TEST_DOWNLOAD"), reason="network; set FOAVIS_TEST_DOWNLOAD=1")
def test_download_one(tmp_path):
    download_checkpoints(str(tmp_path), ["cmnist_learned_s1"])
    assert os.path.getsize(tmp_path / "cmnist_learned_s1.pth") > 0
