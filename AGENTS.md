# Notes for working on this code

Code of the FoAViS paper (see `README.md` for the paper, installation and data, and `docs/` for evaluation, training and the paper-to-code map). This file is the
short orientation for anyone changing the code, people and AI coding assistants alike.

## Map

One forward pass (`foavis/model/foavis.py`, `FoAViS.forward`), for t = 1..T, starting at the image centre:

```
glimpse   = foveation.glimpse(image, f_t)                   foveation.py   log-polar sampling, 64 x 64
features  = foveation.backproject(backbone(glimpse), G)     backbone.py    Res-UNet; back to a G x G image grid
H_t       = memory(features, H_{t-1})                       memory.py      LSTM cell per grid position
logits    = policy_head(sg[H_t])                            heads.py       -> pi(. | H_t) over the G x G cells
f_{t+1}   = sampler(logits) or a fixed / heuristic policy   policy.py
y_t       = classification_head(H_t) | segmentation_head(H_t)
```

| Where | What |
|---|---|
| `main.py` | config, data, model, checkpoint loading, optimizer and schedule, then `Trainer` |
| `foavis/model/passive.py` | passive baselines (`model.passive_input_size > 0`): backbone and task head on the whole image, same `Outputs` with T = 1 |
| `foavis/core/trainer.py` | training epochs, evaluation passes, metrics, checkpoints, scanpath recording and replay |
| `foavis/core/losses.py` | task loss on every step, reward, reward standardization, policy loss (paper Eq. 5-9) |
| `foavis/data/` | datasets (`cmnist.py`, `imagenet.py`, `segmentation.py`), `transforms.py`, `loaders.py` |
| `foavis/io/` | `checkpoint.py`, `scanpaths.py` (Parquet), `download.py` (released checkpoints), `logger.py` |
| `foavis/predict.py` | single-image inference for the released models (used by `demo.ipynb`) |
| `foavis/utils/parser.py` | config loading and command-line flags |
| `configs/config_default.yaml` | every config key and its default; `configs/paper/*.yaml` only override it |
| `reproduce/` | Tables 2 and 3: `table2.sh`, `table3.sh` (via `common.sh`) -> `cells.csv` -> `aggregate.py` |

## Conventions

- Fixations are (x, y) in [-1, 1] image coordinates, (-1, -1) top left. `Outputs.fixations` is (B, T+1, 2): the
  last entry is proposed but never fixated.
- `Outputs` fields are stacked over steps: `class_logits` (B, T, classes), `policy_logits` (B, T, G, G),
  `log_pis`, `entropies` (B, T). In training, `seg_logits` holds only the last step and `seg_logits_early` the others.
- Configs: a key that is not in `config_default.yaml` is an error (in run configs and on the command line), and
  reading a missing key raises. A new option goes into `config_default.yaml` with a comment, default off/unchanged.
- `batch_size` is per GPU. Under torchrun, only the main process (`dist_info['is_main_process']`) logs, saves
  checkpoints and writes scanpaths.
- Evaluation (`--just_evaluate true`) takes the argmax of a learned policy unless `--policy_argmax` is given, runs
  `training.eval_passes` passes over the validation set and reports the last one (`Validation Metrics: {...}`).

## Invariants

The released checkpoints and `reproduce/expected.csv` pin the behaviour. A change must not alter:

- parameter names and shapes (module attribute names are checkpoint keys: renaming `self.backbone` breaks loading);
- the random-number draws of the C-MNIST generator (validation images are a function of the index) and the
  transforms;
- the evaluation protocol above, and the 4-decimal rounding of recorded fixations (replayed cells depend on it);
- the policy-loss scaling (`REFERENCE_DECISION_STEPS`) and the class weights of the segmentation loss.

## Checking a change

```bash
python -m pytest tests/                       # CPU tests in seconds; with a GPU and DATA_DIR also a training epoch
python main.py --config_path configs/paper/cmnist_learned.yaml --just_evaluate true \
    --pretrained_model_path checkpoints/cmnist_learned_s1.pth   # 'val_accuracy': 99.33
OUT=/tmp/check bash reproduce/table2.sh cmnist && python reproduce/aggregate.py /tmp/check/cells.csv
```

`aggregate.py` compares every evaluated cell with `expected.csv` (tolerance 0.05 for deterministic policies, 0.5
for random fixations). Pitfalls when comparing runs:

- The validation loader shuffles, so `--use_test_mode` (at most 100 batches) sees a different subset whenever the
  random-number stream before it changes. Compare full validation passes.
- GPU kernels are not deterministic: identical code can differ by ~0.01 in a metric.
- A recorded scanpath file must cover every validation image it is replayed on.
