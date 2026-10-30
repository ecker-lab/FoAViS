import glob
import os

import pytest
import torch

from foavis.core.losses import LossCalculator
from foavis.model.foavis import build_model
from foavis.utils.parser import get_args_and_cfg

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(REPO, "configs", "config_default.yaml")


def cfg(*argv):
    return get_args_and_cfg(DEFAULT, list(argv))


def test_flags_by_name_and_by_path():
    c = cfg("--num_glimpses", "6", "--memory.channels", "32", "--lr", "1e-4", "--use_test_mode")
    assert c.model.num_glimpses == 6 and c.model.memory.channels == 32
    assert c.training.optimizer.lr == 1e-4 and isinstance(c.training.optimizer.lr, float)
    assert c.utils.use_test_mode is True


def test_ambiguous_and_unknown_flags_fail():
    with pytest.raises(KeyError):
        cfg("--channels", "32")          # in backbone, memory and policy_head
    with pytest.raises(KeyError):
        cfg("--no_such_key", "1")


def test_evaluation_uses_argmax_unless_given():
    assert cfg("--just_evaluate", "true").model.policy_argmax is True
    assert cfg("--just_evaluate", "true", "--policy_argmax", "false").model.policy_argmax is False


def test_policy_loss_needs_learned_policy_for_training():
    with pytest.raises(ValueError):
        cfg("--policy", "random")
    cfg("--policy", "random", "--policy_loss.enabled", "false")


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(REPO, "configs", "paper", "*.yaml"))))
def test_paper_configs_load(path):
    cfg("--config_path", path)


def small_model(task="classification", *argv):
    c = cfg("--dataset_name", {"classification": "imagenet", "semantic_seg": "ade20k_semantic_seg"}[task], "--backbone.channels", "8", "--decoder_channels", "16", "--memory.channels", "16",
            "--grid_size", "16", "--policy_head.channels", "8", "--num_glimpses", "3", *argv)
    c.model.num_classes, c.model.image_channels = 5, 1
    return c, build_model(c, image_size=32)


@pytest.mark.parametrize("task,policy", [("classification", p) for p in ("learned", "random", "triangle", "saliency")]
                         + [("semantic_seg", p) for p in ("learned", "entropy")])
def test_forward_shapes(task, policy):
    argv = ["--policy", policy] + ([] if policy == "learned" else ["--policy_loss.enabled", "false"])
    argv += ["--num_glimpses", "4"] if policy == "triangle" else []
    c, model = small_model(task, *argv)
    T = c.model.num_glimpses
    out = model(torch.randn(2, 1, 32, 32), training=False)
    assert out.fixations.shape == (2, T + 1, 2) and out.policy_logits.shape == (2, T, 16, 16)
    assert out.fixations.abs().max() <= 1
    if task == "classification":
        assert out.class_logits.shape == (2, T, 5)
    else:
        assert out.seg_logits.shape == (2, 1, 5, 32, 32) and out.seg_predictions.shape == (2, T, 32, 32)


def test_stop_grad_keeps_policy_gradient_in_the_policy_head():
    c, model = small_model()
    c.model.policy_loss.weight = 1.0
    out = model(torch.randn(4, 1, 32, 32), training=True)
    loss = LossCalculator(c).policy_loss(out.log_pis, out.entropies, torch.randn(4, c.model.num_glimpses))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.policy_head.parameters())
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(p.grad is None for p in model.memory.parameters())


def passive_model(dataset, *argv):
    c = cfg("--dataset_name", dataset, "--passive_input_size", "32", "--policy_loss.enabled", "false",
            "--backbone.channels", "8", "--decoder_channels", "16", *argv)
    c.model.num_classes, c.model.image_channels = 5, 1
    return c, build_model(c, image_size=48)


def test_passive_baseline_shapes():
    _, model = passive_model("imagenet")
    assert model(torch.randn(2, 1, 48, 48), training=False).class_logits.shape == (2, 1, 5)
    _, model = passive_model("ade20k_semantic_seg")
    out = model(torch.randn(2, 1, 48, 48), training=False)
    assert out.seg_logits.shape == (2, 1, 5, 48, 48) and out.seg_predictions.shape == (2, 1, 48, 48)


def test_passive_segmentation_initialised_from_classification(tmp_path):
    from foavis.io.checkpoint import load_checkpoint
    _, cls = passive_model("imagenet")
    torch.save({"model_state_dict": cls.state_dict()}, tmp_path / "cls.pth")
    _, seg = passive_model("ade20k_semantic_seg")
    load_checkpoint(seg, str(tmp_path / "cls.pth"))
    for k, v in cls.backbone.state_dict().items():
        assert torch.equal(seg.backbone.state_dict()[k], v)


def test_passive_baseline_needs_no_policy_loss():
    with pytest.raises(ValueError):
        cfg("--passive_input_size", "128")
