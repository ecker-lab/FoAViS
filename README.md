# FoAViS: Analyzing Learned Fixation Strategies in a Foveated Active Vision System

Human vision is foveated and active: resolution is highest at the fovea and falls off towards the periphery, while fixations are actively directed to locations of interest. Building a machine system that processes information in the same way lets us study gaze behavior in a controlled setting where the system's components and objectives are known. The emergence of phenomena such as inhibition of return or task-specific fixation patterns can therefore be examined systematically, which may in turn inform our understanding of biological vision. This requires a model that can sample like the fovea, handle various tasks from classification to dense prediction, and be trained concurrently with a fixation policy. While prior work on active vision typically addresses only a subset of these aspects, a unified framework would be desirable to simplify model training and investigate gaze behavior under different task constraints. In our study, we present such a system and study its capabilities and the fixation policies that emerge from it. Our Foveated Active Vision System (FoAViS) samples visual input by adopting the established log-polar foveation and accumulates information in a 2D spatial memory. Due to its 2D structure, the same memory can be read out globally for classification or densely for semantic segmentation, without any architectural change. Fixations are drawn from a 2D map by a policy trained concurrently with the task. We find that learned fixation policies select more informative, task-specific locations than non-learned heuristics. These policies also shape the task representations in a beneficial way: for both classification and segmentation, models trained concurrently with a policy outperform models trained with random locations, even when the latter are provided with learned scanpaths. Moreover, these scanpaths exhibit similarities to human gaze behavior by capturing aspects of empirical scanpath statistics and suppressing re-fixation of recently attended locations. Our work marks an important step away from standard passive, single-pass architectures, towards biologically grounded models of vision. It opens a path towards more comprehensive, biologically plausible models for studying goal-directed gaze behavior and information representation in the visual system.

The paper is on [arXiv](https://arxiv.org/abs/2610.41987). The pre-trained models are on
[GRO.data](https://doi.org/10.25625/UXZTPA), and the code downloads them automatically if required.

## Method overview

<p align="center"><img src="docs/architecture.png" width="760" alt="FoAViS unrolled over three glimpses"></p>

**FoAViS unrolled over three glimpses.** At each step, the current fixation serves as the center for a foveation module
that extracts a log-polar glimpse (1). The glimpse is encoded by the backbone (2), backprojected to Cartesian image
space, and accumulated into a 2D spatial memory using a shared per-position LSTM (3). The memory state is then read out
by the policy head (4) to produce a spatial probability map from which the next fixation is drawn (purple number), while
the task head (5) uses it to generate the task prediction. The sampling grid is shown in white, feature maps and memory
states are visualized via PCA, and the policy logits as a heatmap.

## Demo

The demo notebook visualizes scanpaths of our pre-trained models. It runs on Google
Colab without any setup:

<a target="_blank" href="https://colab.research.google.com/github/ecker-lab/FoAViS/blob/main/demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Repository overview

This repository contains the code to train and evaluate FoAViS on the four datasets used in our paper (Cluttered MNIST, ImageNet,
ADE20K and COCO-Stuff). For details, we refer to the following guides:

- To visualize scanpaths of the pre-trained models, use the [demo notebook](demo.ipynb) or
  the [inference guide](docs/inference.md).
- To evaluate a model, reproduce the paper's results, or record the fixations a model
  makes, see the [evaluation guide](docs/evaluation.md).
- To train a model, see the [training guide](docs/training.md).
- To find your way around the code, see the [code guide](docs/code.md). It maps parts of our paper to the code.

Training and evaluation on the full datasets require a GPU, but running a model on a single image also works on a CPU.

## Installation

To install the required dependencies, we recommend using conda:

```bash
conda env create -f environment.yml
conda activate foavis
pip install -e .
```

The environment uses Python 3.11 and PyTorch 2.3.0 with CUDA 12.1, so your NVIDIA driver must support CUDA 12.1. Other Python and PyTorch versions may work as well but have not been tested. The code was tested on Linux.

## Data

All datasets must be placed under one root directory, which you pass to every command as `--data_dir` (default `./data`).

Cluttered MNIST (C-MNIST) needs no preparation. The code generates its images on the fly, each with one MNIST digit
(the target) among five EMNIST letters (the distractors) on a 224 x 224 canvas, and downloads MNIST and EMNIST the
first time they are used.

You have to download the other datasets yourself from their official sources and arrange them as shown below:
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

Run the pre-trained ImageNet model on an example image. It prints the fixations and the five most likely classes,
and saves a figure of where the model looked to `docs/examples/cat_foavis.png`:

```bash
python -m foavis.predict docs/examples/cat.jpg --model imagenet_learned_s1
```

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
