# Evaluation

How to evaluate a FoAViS model on a validation set, whether one of our pre-trained models or one you trained
yourself, how to reproduce the paper's main results, and how to record the fixations a model makes. Set up the
environment and the data first, as described in the [README](../README.md).

- [Pre-trained models](#pre-trained-models)
- [Evaluating a model](#evaluating-a-model)
- [Reproducing the paper's results](#reproducing-the-papers-results)
- [Recording and replaying fixations](#recording-and-replaying-fixations)

## Pre-trained models

For each dataset we release two models. The `learned` model was trained with its own fixation policy, and the
`random` model, which the paper compares it with, was trained with random fixation locations. The paper trains every
configuration with three seeds, and the released models are the first seed. They are hosted on [GRO.data](https://doi.org/10.25625/UXZTPA) and named
`<dataset>_<policy>_s1.pth`, with `<dataset>` one of `cmnist, imagenet, ade20k, coco_stuff` and `<policy>` one of
`learned, random`.

Download all eight (about 0.45 GB) or only the ones you need into `./checkpoints`:

```bash
python -m foavis.io.download                                     # all
python -m foavis.io.download --models imagenet_learned_s1        # a subset
```

The download checks each file against its MD5. The scripts in `reproduce/` download missing checkpoints by
themselves.

Loading `imagenet_random_s1` prints a warning that some policy-head parameters are missing or have another shape.
The random-location models never use their policy head, and this one was saved with a smaller head than the config
builds, so the warning does not affect any result.

## Evaluating a model

To evaluate a model, run `main.py --just_evaluate true` with the config it was trained with and its checkpoint:

```bash
python main.py --config_path configs/paper/ade20k_learned.yaml --data_dir <data_dir> --just_evaluate true \
    --pretrained_model_path checkpoints/ade20k_learned_s1.pth
```

The end of the log shows the validation metrics: accuracy for classification, mIoU and pixel accuracy for
segmentation. For a model you trained yourself, use the `config.yaml` and a checkpoint from its run directory
(see the [training guide](training.md)).

These options change the evaluation:

| Option | Effect |
|---|---|
| `--num_glimpses T` | number of glimpses per image (the models are trained with 4) |
| `--policy <name>` | where fixations come from: `learned` (the policy head), `random`, the fixed sequences `triangle` and `quadrant`, or the map heuristics `saliency` (classification) and `entropy` (segmentation) |
| `--ior_radius r` | for `saliency` and `entropy`: exclude a radius around earlier fixations (inhibition of return; the paper uses 0.5) |
| `--policy_argmax false` | sample from the learned policy instead of taking its most likely location (the default in evaluation) |
| `--batch_size B` | evaluation batch size |

## Reproducing the paper's results

The scripts in `reproduce/` evaluate every model of the paper's two main results tables and print the tables:

- `table2.sh`: each model evaluated with the policy it was trained with, at 1, 4 and 6 glimpses;
- `table3.sh`: the models evaluated with other fixation policies: random locations, fixed sequences, map
  heuristics and the learned policies of other datasets (policy transfer).

```bash
bash reproduce/table2.sh                     # all datasets, or e.g. `bash reproduce/table2.sh cmnist imagenet`
bash reproduce/table3.sh
python reproduce/aggregate.py                # prints both tables and compares them with the paper's values
```

`aggregate.py` checks every value against `reproduce/expected.csv`, which holds the values of the released models.
Deterministic policies (the learned policy, the fixed sequences and the map heuristics) match up to GPU
non-determinism, within 0.05. Random-location evaluations draw new fixations every time, so they match within 0.5. The paper reports the
mean and standard deviation over three seeds. If you trained the other seeds yourself, put them next to the
released ones as `<dataset>_<policy>_s<seed>.pth` and run the scripts with `SEEDS="1 2 3"`.

Each table entry is one evaluation run on one GPU. Logs go to `reproduce_out/logs/` and results to
`reproduce_out/cells.csv`. A rerun skips finished entries, so an interrupted run continues where it stopped, and you
can run separate datasets in parallel. The scripts read the environment variables `CKPT_DIR`
(default `./checkpoints`), `DATA_DIR` (default `./data`), `OUT` (default `./reproduce_out`), `BS` (default 8) and
`SEEDS` (default 1).

We do not release weights for the passive baselines in the first row of the paper's main table. To reproduce them,
train them with the `*_passive` configs (see the [training guide](training.md)).

## Recording and replaying fixations

To record where a model looks, add `--save_scanpaths <dir>` to an evaluation. It writes `<dir>/scanpaths.parquet`
with one row per validation image: the image path and the T+1 fixations, as a flat list `[x_1, y_1, x_2, y_2, ...]`
in normalized image coordinates, where `(-1, -1)` is the top left and `(1, 1)` the bottom right corner.

```bash
python main.py --config_path configs/paper/imagenet_learned_phase2.yaml --data_dir <data_dir> --just_evaluate true \
    --pretrained_model_path checkpoints/imagenet_learned_s1.pth --save_scanpaths scanpaths/imagenet_learned
```

```python
import pyarrow.parquet as pq
table = pq.read_table("scanpaths/imagenet_learned/scanpaths.parquet").to_pydict()   # image_path, epoch, locations
```

With `--replay_scanpaths <file>`, any model looks at the recorded locations instead of choosing its own. `table3.sh`
uses this to transfer a learned policy to another model. Images are matched by class folder and file name, so the
data root can differ between recording and replay.
