# Inference

This guide covers running a pre-trained FoAViS model on single images, for example your own photos, and looking at
where it fixates. For evaluating a model on a whole validation set, see the [evaluation guide](evaluation.md). The
[demo notebook](../demo.ipynb) shows everything below with example outputs and runs on Google Colab.

## Command line

```bash
python -m foavis.predict my_photo.jpg --model imagenet_learned_s1
```

prints the fixations and the prediction and saves a figure to `my_photo_foavis.png` (`--out` sets another path). The
figure shows the image with the numbered fixations, the policy map after each glimpse (from which the next fixation
is drawn) and the prediction. The model's checkpoint is downloaded on first use. `--glimpses T` changes the number of
glimpses (the models are trained with 4).

Which model to use depends on the image:

| Model | Input | Prediction |
|---|---|---|
| `imagenet_learned_s1` | a photo of one object | one of the 1000 ImageNet classes |
| `ade20k_learned_s1` | a scene, indoors or outdoors | a label for every pixel, 150 classes |
| `coco_stuff_learned_s1` | a scene | a label for every pixel, 182 classes |

The `*_random_s1` models were trained with random fixation locations and also fixate randomly. The C-MNIST models
only work on C-MNIST images (see below).

## Python

```python
from foavis.predict import load_model, predict, show

model = load_model("ade20k_learned_s1")            # also: num_glimpses=8, policy="random", device="cpu"
result = predict(model, "my_photo.jpg")            # a path, a PIL image or a preprocessed tensor
result.fixations                                   # (T, 2), the first at the image centre
result.classes_present()                           # segmentation: [(class, fraction of pixels), ...]
result.top(5)                                      # classification: [(class, probability), ...]
show(result, "figure.png")
```

`policy` takes any of the evaluation policies: `learned`, `random`, `triangle`, `quadrant`, `saliency`
(classification) and `entropy` (segmentation). The result also holds the per-glimpse predictions (`result.steps`),
which show how the prediction changes as the model sees more of the image, and the policy logits after each glimpse
(`result.policy_logits`).

The image is preprocessed as in the paper's evaluation: resized to 256 pixels and centre-cropped to 224 for ImageNet,
resized to 512 x 512 for the segmentation models. Fixations are in normalized coordinates of that model input,
`(x, y)` with `(-1, -1)` the top left and `(1, 1)` the bottom right corner.

C-MNIST images are generated, so the C-MNIST models are run on a validation image:

```python
from foavis.predict import cmnist_example
image, digit = cmnist_example(index=0, data_dir="./data")   # downloads MNIST and EMNIST on first use
result = predict(load_model("cmnist_learned_s1"), image)
```
