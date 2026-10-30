"""Recorded scanpaths (Parquet): written by `--save_scanpaths <dir>`, replayed by `--replay_scanpaths <file>`.

One row per image and evaluation pass: image_path (str), epoch (the pass, int16), locations (the T+1
fixations, flat [x_1, y_1, x_2, y_2, ...] in [-1, 1] image coordinates, rounded to 4 decimals).
Pass p of a replay run reads the fixations recorded in pass p. Images are matched on the last two path
components (class folder / file name), so the data root may differ between recording and replay.
"""
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FILE_NAME = "scanpaths.parquet"


def image_key(path):
    parts = [p for p in os.path.normpath(path).split(os.sep) if p not in ("", ".")]
    return os.sep.join(parts[-2:])


class ScanpathWriter:
    def __init__(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, FILE_NAME)
        self.rows = {"image_path": [], "epoch": [], "locations": []}

    def write_batch(self, image_paths, fixations, epoch):
        """fixations: (B, T+1, 2) numpy array."""
        self.rows["image_path"] += list(image_paths)
        self.rows["epoch"] += [epoch] * len(image_paths)
        self.rows["locations"] += [np.round(f.flatten(), 4).astype(np.float32).tolist() for f in fixations]

    def finalize(self):
        schema = pa.schema([("image_path", pa.string()), ("epoch", pa.int16()), ("locations", pa.list_(pa.float32()))])
        pq.write_table(pa.table(self.rows, schema=schema).sort_by("image_path"), self.path, compression="snappy")
        print(f"wrote {len(self.rows['image_path'])} scanpaths to {self.path}")


class ScanpathReader:
    def __init__(self, path):
        table = pq.read_table(os.path.join(path, FILE_NAME) if os.path.isdir(path) else path,
                              columns=["image_path", "epoch", "locations"]).to_pydict()
        self.passes = {}   # pass index -> {image key: (T+1, 2)}
        epochs = sorted(set(table["epoch"]))
        for path, epoch, flat in zip(table["image_path"], table["epoch"], table["locations"]):
            self.passes.setdefault(epochs.index(epoch), {})[image_key(path)] = \
                np.asarray(flat, dtype=np.float32).reshape(-1, 2)

    def get_locations(self, eval_pass, image_paths):
        """(B, T+1, 2) fixations of these images in the given evaluation pass."""
        if eval_pass not in self.passes:
            raise IndexError(f"evaluation pass {eval_pass} was not recorded ({len(self.passes)} passes)")
        lookup = self.passes[eval_pass]
        missing = [p for p in image_paths if image_key(p) not in lookup]
        if missing:
            raise KeyError(f"{len(missing)} images have no recorded scanpath, e.g. {missing[0]}")
        return np.stack([lookup[image_key(p)] for p in image_paths])
