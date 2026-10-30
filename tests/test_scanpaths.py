import numpy as np
import pytest

from foavis.io.scanpaths import ScanpathReader, ScanpathWriter


def test_record_and_replay(tmp_path):
    rng = np.random.default_rng(0)
    paths = ["/data/ImageNet/val/n01/a.JPEG", "/data/ImageNet/val/n02/b.JPEG", "17"]
    fixations = {p: rng.uniform(-1, 1, size=(3, 5, 2)) for p in range(2)}   # pass -> (B, T+1, 2)
    writer = ScanpathWriter(str(tmp_path))
    for p in range(2):
        writer.write_batch(paths, fixations[p], epoch=p)
    writer.finalize()
    reader = ScanpathReader(str(tmp_path))   # the directory or the file
    # another data root, other order
    got = reader.get_locations(1, ["/other/root/val/n02/b.JPEG", "17", "/data/ImageNet/val/n01/a.JPEG"])
    np.testing.assert_allclose(got, fixations[1][[1, 2, 0]], atol=5e-5)
    with pytest.raises(KeyError):
        reader.get_locations(0, ["/data/ImageNet/val/n03/c.JPEG"])
    with pytest.raises(IndexError):
        reader.get_locations(2, paths)
