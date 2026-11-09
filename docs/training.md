# Training

This guide shows how to train the paper's FoAViS models or your own variants of them. Set up the environment and the data first, as
described in the [README](../README.md).

- [Running a training](#running-a-training)
- [The paper's models](#the-papers-models)
- [Training from scratch](#training-from-scratch)
- [Starting from a pre-trained model](#starting-from-a-pre-trained-model)
- [Resuming an interrupted run](#resuming-an-interrupted-run)
- [Additional models of the appendix](#additional-models-of-the-appendix)

## Running a training

Every model of the paper has a config in `configs/paper/`. The header of each config gives the exact command, the
hardware of the reported runs and their seeds. The body lists only what differs from `configs/config_default.yaml`,
which documents every option. Start a training with

```bash
python main.py --config_path configs/paper/<model>.yaml --data_dir <data_dir> --seed <seed>     # one GPU
torchrun --standalone --nproc_per_node 4 main.py --config_path configs/paper/<model>.yaml ...   # four GPUs
```

`batch_size` and `num_workers` are per GPU. With a different number of GPUs, keep the total batch (GPUs x
`batch_size`) as in the table below, e.g. ImageNet on 2 GPUs with `--batch_size 32`. The learned models standardize
their rewards within each GPU's batch, so a different batch size per GPU might lead to slightly different training
behavior, unless the standardization is changed to use the total batch across GPUs.

Any config option can be overridden on the command line by its name (`--num_glimpses 6`) or, where the name occurs in
several sections, by its path (`--memory.channels 64`). Each run writes its config, its checkpoints and a link
`last.pth` to the latest checkpoint into `model_ckpts/<dataset>/<timestamp>_<id>/`. Every new best checkpoint is kept,
so a run folder can grow to several GB (about 6 GB for a segmentation run). Metrics go to the console; add
`--use_wandb true` to also log them to Weights & Biases.

## The paper's models

| Config | Starts from | Epochs | GPUs x batch per GPU | Memory per GPU | Time per run |
|---|---|---|---|---|---|
| `cmnist_random`, `cmnist_learned` | scratch | 400 | 1 x 64 | ~11 GB | ~20 h, A100 |
| `imagenet_*_phase1` | scratch | 75 (constant learning rate) | 4 x 16 | ~5 GB | ~35 h, 4x A100 or H100 |
| `imagenet_*_phase2` | phase 1 of the same model | 75 (step decay) | 4 x 16 | ~5 GB | ~35 h, 4x A100 or H100 |
| `ade20k_random`, `ade20k_learned` | ImageNet random-location model | 500 | 1 x 16 | > 40 GB (~51 GB learned) | 44-50 h, H100 |
| `coco_stuff_random`, `coco_stuff_learned` | ImageNet random-location model | 72 | 1 x 16 | > 40 GB | 43-49 h, H100 |
| `cmnist_passive`, `imagenet_passive` | scratch | 400, 150 | 1 x 64, 4 x 16 | ~10 GB, < 40 GB | ~9 h, A100; ~35 h, 4x A100 |
| `ade20k_passive`, `coco_stuff_passive` | `imagenet_passive` | 500, 72 | 1 x 16 | < 40 GB | ~20 h, ~19 h, A100 |

`random` models were trained with random fixation locations, `learned` models with their own fixation policy, and
the `passive` models are the baselines without foveation. Training is not bit-reproducible on GPUs, so a retrained model
matches the reported one up to seed variation.

The times are from runs on the hardware available at that time. On an 80 GB A100, for example, the
segmentation models with foveation took about 1.5 times as long as on the H100s listed.
Memory per GPU was read from `nvidia-smi` while training on A100s with the listed batch size. "> 40 GB" means that the
run did not fit on a 40 GB A100, so the segmentation models with foveation need an 80 GB GPU at batch 16.

## Training from scratch

The C-MNIST models and the passive C-MNIST and ImageNet baselines train from scratch in one run:

```bash
python main.py --config_path configs/paper/cmnist_learned.yaml --data_dir <data_dir> --seed <seed>
```

The ImageNet models train in two phases of 75 epochs. Phase 1 uses a constant learning rate. Phase 2 continues from
the end of phase 1, with its optimizer state, and decays the learning rate:

```bash
torchrun --standalone --nproc_per_node 4 main.py --config_path configs/paper/imagenet_learned_phase1.yaml --data_dir <data_dir> --seed <seed>
torchrun --standalone --nproc_per_node 4 main.py --config_path configs/paper/imagenet_learned_phase2.yaml --data_dir <data_dir> --seed <seed> \
    --pretrained_model_path model_ckpts/imagenet/<phase-1 run>/last.pth
```

## Starting from a pre-trained model

Both the `random` and the `learned` segmentation models (ADE20K, COCO-Stuff) start from the ImageNet model trained
with random locations. You can use the released one instead of training it yourself:

```bash
python -m foavis.io.download --models imagenet_random_s1
python main.py --config_path configs/paper/ade20k_learned.yaml --data_dir <data_dir> --seed <seed> \
    --pretrained_model_path checkpoints/imagenet_random_s1.pth
```

The passive segmentation baselines start in the same way from a trained `imagenet_passive` model. More generally,
`--pretrained_model_path` initializes a model from any checkpoint. Weights whose shape does not fit, such as a task
head for another number of classes, are skipped and listed in the log.

## Resuming an interrupted run

Rerun the same command with

```bash
--pretrained_model_path model_ckpts/<dataset>/<run>/last.pth --resume_training true
```

This restores the optimizer, the learning-rate schedule and the epoch, and continues training in a new run folder.
Its `best.pth` covers only the epochs after the resume.

## Additional models of the appendix

Each config header names what differs from the corresponding main model.

| Configs | Appendix experiment |
|---|---|
| `*_learned_no_stopgrad`, `*_no_stopgrad_small_head`, `*_no_stopgrad_entropy` (4 datasets) | stop-gradient ablation |
| `ade20k_learned_no_stopgrad_lambda001` | stop-gradient ablation, policy-loss weight 0.01 |
| `imagenet_policy_only_{terminal,perstep}` | reward-horizon comparison (only the policy head is trained, on the frozen ImageNet random-location model) |

The ImageNet runs of the stop-gradient ablation correspond to phase 1 (75 epochs, constant learning rate) and are
compared with `imagenet_learned_phase1`. The comparison of per-step and last-step task supervision was run with an
earlier version of the code and has no config here.
