# FoAViS: Analyzing Learned Fixation Strategies in a Foveated Active Vision System

Human vision is foveated and active: resolution is highest at the fovea, and fixations are directed to locations of
interest. FoAViS is a machine vision system that works the same way. It looks at an image through a sequence of
log-polar glimpses, accumulates what it sees in a 2D spatial memory, and decides where to look next with a fixation
policy that is trained together with the task. Because the memory is spatial, the same model can be read out
globally for classification or densely for semantic segmentation. In the paper we use it to study which fixation
strategies emerge and how they shape the task representations.

The paper is available on [arXiv](https://arxiv.org/abs/2610.41987) (under review at TMLR). The pre-trained models
are available on [GRO.data](https://doi.org/10.25625/UXZTPA); this repository downloads them for you, see
[Quickstart](#quickstart).

<p align="center"><img src="docs/architecture.png" width="760" alt="FoAViS unrolled over three glimpses"></p>

*FoAViS unrolled over three glimpses. At each step the model samples a log-polar glimpse around the current
fixation (1), encodes it with a Res-UNet backbone (2), maps the features back to image coordinates and accumulates
them in a 2D spatial memory with an LSTM cell per position (3). A policy head turns the memory into a probability
map from which the next fixation is drawn (4), and a task head reads the memory out for classification or semantic
segmentation (5).*

## Repository overview

This repository contains the code to train and evaluate FoAViS on the four datasets of the paper (Cluttered MNIST,
ImageNet, ADE20K and COCO-Stuff), together with the configurations of the paper's models, their pre-trained weights
and scripts that reproduce the paper's main results. Depending on what you want to do:

- **Evaluate** the pre-trained models or your own, reproduce the paper's results, or record the fixations a model
  makes: see the [evaluation guide](docs/evaluation.md).
- **Train** a model, from scratch or starting from a pre-trained one: see the [training guide](docs/training.md).
- **Understand or change the code**: the [code guide](docs/code.md) maps every part of the paper to the code and
  explains how to run the tests.

All of this needs a GPU. The code has been tested on Linux.

## Installation

We recommend conda:

```bash
conda env create -f environment.yml
conda activate foavis
pip install -e .
```

The environment uses Python 3.11 and PyTorch 2.3.0 with CUDA 12.1, so your NVIDIA driver must support CUDA 12.1.

## Data

All datasets live under one root directory, which you pass to every command as `--data_dir` (default `./data`).

**Cluttered MNIST (C-MNIST) needs no preparation.** Its images are generated on the fly: each places one MNIST
digit (the target) among five EMNIST letters (the distractors) on a 224 x 224 canvas. MNIST and EMNIST are downloaded
automatically on first use.

**The other datasets have to be downloaded by you** from their official sources and arranged as below:
[ImageNet](https://image-net.org/download.php) (ILSVRC 2012), [ADE20K](http://sceneparsing.csail.mit.edu/)
(ADEChallengeData2016), the [COCO 2017](https://cocodataset.org/#download) images and the
[COCO-Stuff 164k](https://github.com/nightrome/cocostuff) label maps (stuffthingmaps). You only need the datasets you
want to use.

```
<data_dir>/
  MNIST/, EMNIST/                          created automatically
  ImageNet/
    train_images/<wnid>/*.JPEG
    val/<wnid>/*.JPEG                      validation images sorted into class folders
  ADEChallengeData2016/
    images/{training,validation}/*.jpg
    annotations/{training,validation}/*.png
  coco2017/{train2017,val2017}/*.jpg
  coco_stuff/{train2017,val2017}/*.png     COCO-Stuff label maps (stuffthingmaps)
```

## Quickstart

Download the pre-trained C-MNIST model and evaluate it on the validation set:

```bash
python -m foavis.io.download --models cmnist_learned_s1
python main.py --config_path configs/paper/cmnist_learned.yaml --just_evaluate true \
    --pretrained_model_path checkpoints/cmnist_learned_s1.pth
```

The log ends with `'val_accuracy': 99.33`. To reproduce the C-MNIST column of the paper's main results table:

```bash
bash reproduce/table2.sh cmnist && python reproduce/aggregate.py
```

## Citation

If you find this work useful, please cite:

```bibtex
@misc{foavis,
  title         = {{FoAViS}: Analyzing Learned Fixation Strategies in a Foveated Active Vision System},
  author        = {Hassler, Valentin and Henrich, Jonathan and Berreby, Yoha{\"i}-Eliel and Zhu, Haoze and Ecker, Alexander},
  year          = {2026},
  eprint        = {2610.41987},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2610.41987}
}
```

## License

MIT, see `LICENSE`.
