# FoAViS: Analyzing Learned Fixation Strategies in a Foveated Active Vision System

Human vision is foveated and active: resolution is highest at the fovea and falls off towards the periphery, and
fixations are actively directed to locations of interest. Building a machine system that processes information in the
same way lets us study gaze behavior in a controlled setting where the system's components and objectives are known.
The emergence of gaze characteristics can therefore be examined systematically, which may in turn inform our
understanding of biological vision. This requires a model that can sample like the fovea, handle various tasks from
classification to dense prediction, and be trained concurrently with a fixation policy. In our study, we present such
a system and study its capabilities and the fixation policies that emerge from it. Our system, FoAViS, samples visual
input by adopting the established log-polar foveation and accumulates information in a 2D spatial memory. Due to its
2D structure, the same memory can be read out globally for classification or densely for semantic segmentation,
without any architectural change. Fixations are drawn from a 2D map by a policy trained concurrently with the task. We
find that learned fixation policies select more informative, task-specific locations than non-learned heuristics.
These policies also shape the task representations in a beneficial way: for both classification and segmentation,
models trained concurrently with a policy outperform models trained with random locations, even when the latter are
provided with learned scanpaths. Moreover, these scanpaths exhibit similarities to human gaze behavior by capturing
aspects of empirical scanpath characteristics. Our work marks an important step away from standard passive,
single-pass architectures, towards biologically grounded models of vision. It opens a path for studying goal-directed
gaze behavior and information representation in the visual system.

The paper is available on [arXiv](https://arxiv.org/abs/2610.41987) (under review at TMLR). The pre-trained models
are available on [GRO.data](https://doi.org/10.25625/UXZTPA); this repository downloads them for you, see
[Quickstart](#quickstart).

<p align="center"><img src="docs/architecture.png" width="760" alt="FoAViS unrolled over three glimpses"></p>

**Overview of FoAViS unrolled over three glimpses.** At each step, the current fixation serves as the center for a
foveation module that extracts a log-polar glimpse (1) (sampling grid shown in white). The glimpse is encoded by the
backbone (2) (feature map visualized via PCA), backprojected to Cartesian image space, and accumulated into a 2D
spatial memory using a shared per-position LSTM (3) (memory state visualized via PCA). The memory state is then read
out by the policy head (4) to produce a spatial probability map (logits visualized as a heatmap) from which the next
fixation is drawn (purple number), while the task head (5) uses it to generate the task prediction.

## Demo

To see where the pre-trained models look on example images or on your own, open the demo notebook, which runs on
Google Colab without any setup:

<a target="_blank" href="https://colab.research.google.com/github/ecker-lab/FoAViS/blob/main/demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Repository overview

This repository contains the code to train and evaluate FoAViS on the four datasets of the paper (Cluttered MNIST,
ImageNet, ADE20K and COCO-Stuff), together with the configurations of the paper's models, their pre-trained weights
and scripts that reproduce the paper's main results. Depending on what you want to do:

- **Run** a pre-trained model on your own images and look at its fixations: see the [demo notebook](demo.ipynb) or
  the [inference guide](docs/inference.md).
- **Evaluate** the pre-trained models or your own, reproduce the paper's results, or record the fixations a model
  makes: see the [evaluation guide](docs/evaluation.md).
- **Train** a model, from scratch or starting from a pre-trained one: see the [training guide](docs/training.md).
- **Understand or change the code**: the [code guide](docs/code.md) maps every part of the paper to the code and
  explains how to run the tests.

Training and evaluating on the full datasets needs a GPU; running a model on single images also works on a CPU. The
code has been tested on Linux.

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

Run the pre-trained ImageNet model on an example image; it prints the fixations and the top-5 classes and saves a
figure of where the model looked to `docs/examples/cat_foavis.png`:

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
