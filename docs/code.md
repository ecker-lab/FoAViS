# Code guide

This guide is for working with the code: where each part of the paper is implemented, how the repository is
organized, and how to check a change.

## Paper and code

| Paper | Code | Config |
|---|---|---|
| Foveation module, glimpse (Sec. 3.1, Eq. 1) | `foavis/model/foveation.py` `LogPolarFoveation.glimpse` | `model.foveation` |
| Backbone, Res-UNet (Sec. 3.2) | `foavis/model/backbone.py` | `model.backbone` |
| Backprojection to the memory grid (Sec. 3.3) | `LogPolarFoveation.backproject` | `model.memory.grid_size` |
| Memory module, H_t (Sec. 3.3, Eq. 2) | `foavis/model/memory.py` | `model.memory` |
| Policy head psi, policy pi (Sec. 3.4, Eq. 3) | `foavis/model/heads.py` `PolicyHead`, `foavis/model/policy.py` `FixationSampler` | `model.policy_head` |
| Task head phi (Sec. 3.5, Eq. 4) | `ClassificationHead`, `SegmentationHead` in `heads.py` | |
| Episode of T steps, first fixation at the centre (Sec. 3.6) | `foavis/model/foavis.py` `FoAViS.forward` | `model.num_glimpses` |
| Task loss on every step (Eq. 5) | `foavis/core/losses.py` `LossCalculator.classification` / `.segmentation` | |
| Reward r_t (Eq. 6), standardization (Eq. 7) | `LossCalculator._add_policy_loss`, `.policy_loss` | `model.policy_loss.reward`, `.reward_standardization` |
| Policy loss with stop-grad (Eq. 8), total objective (Eq. 9) | `LossCalculator.policy_loss`; stop-grad in `FoAViS.forward` | `model.policy_loss.stop_grad`, `.weight` |
| Learned-policy / random-location models | | `model.policy`, `model.policy_loss.enabled` |
| Passive baselines (Sec. 4, Table 2) | `foavis/model/passive.py` `PassiveBaseline` | `model.passive_input_size` |
| Heuristic policies of Table 3 | `foavis/model/policy.py` | `model.policy`, `model.ior_radius` |
| Policy transfer (replayed scanpaths) | `foavis/io/scanpaths.py` | `utils.save_scanpaths`, `utils.replay_scanpaths` |

## Repository layout

```
main.py                  entry point: training and evaluation (--just_evaluate)
foavis/model/            foavis.py (the glimpse loop), foveation.py, backbone.py, memory.py, heads.py, policy.py,
                         passive.py (the passive baselines)
foavis/core/             losses.py (objective), trainer.py (training and evaluation loop)
foavis/data/             cmnist.py, imagenet.py, segmentation.py (ADE20K, COCO-Stuff), transforms.py, loaders.py
foavis/io/               checkpoint.py, scanpaths.py (recorded fixations), download.py (pre-trained models), logger.py
foavis/utils/            parser.py (config and command-line options), config.py, distributed.py
configs/                 config_default.yaml (every option, commented) and configs/paper/ (one per model)
reproduce/               scripts for the paper's main results (see the evaluation guide)
tests/                   unit tests and a short training test
```

Fixation locations are in normalized image coordinates throughout: `(-1, -1)` is the top left and `(1, 1)` the
bottom right corner of the image.

## Tests

```bash
DATA_DIR=<data_dir> python -m pytest tests/
```

The training test runs one short epoch per model type on a GPU (C-MNIST with learned and random locations, and
ADE20K if it is present under `DATA_DIR`); the other tests run on the CPU in seconds.

If you work on the code with an AI coding assistant, `AGENTS.md` summarizes the structure, the invariants and how to
check a change.
