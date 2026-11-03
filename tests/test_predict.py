import os

import numpy as np
import pytest

from foavis.predict import Prediction, load_model, predict, preprocess, show

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "docs", "examples")


def test_preprocess_sizes():
    assert preprocess(os.path.join(EXAMPLES, "cat.jpg"), "imagenet").shape == (1, 3, 224, 224)
    assert preprocess(os.path.join(EXAMPLES, "living_room.jpg"), "ade20k").shape == (1, 3, 512, 512)


def test_prediction_summaries():
    seg = np.zeros((2, 4, 4), dtype=np.int64)
    seg[-1, :, 1:] = 2
    r = Prediction(image=np.zeros((4, 4, 3)), fixations=np.zeros((2, 2)), policy_logits=np.zeros((2, 3, 3)),
                   steps=seg, task="semantic_seg", policy="learned", class_names=["a", "b", "c"])
    assert r.classes_present() == [("c", 0.75), ("a", 0.25)]
    r.steps, r.task = np.array([[0.1, 0.7, 0.2]]), "classification"
    assert [n for n, _ in r.top(2)] == ["b", "c"]


@pytest.mark.skipif(not os.environ.get("FOAVIS_TEST_DOWNLOAD"), reason="network; set FOAVIS_TEST_DOWNLOAD=1")
def test_predict_example(tmp_path):
    model = load_model("imagenet_learned_s1", device="cpu")
    result = predict(model, os.path.join(EXAMPLES, "cat.jpg"))
    assert result.fixations.shape == (4, 2) and np.all(result.fixations[0] == 0)
    assert result.top(1)[0][0] == "tabby"
    show(result, tmp_path / "cat.png")
    assert (tmp_path / "cat.png").exists()
