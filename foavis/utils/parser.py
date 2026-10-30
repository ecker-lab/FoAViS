"""Configuration: configs/config_default.yaml, overridden by a run config (--config_path) and by command-line flags.

A flag names a config key by its last component when that is unique (--num_glimpses 6), otherwise by enough of
its path to be unique (--memory.channels 128). Values are parsed as YAML. A key that does not exist in the
defaults is an error, in run configs as well as on the command line.
"""
import sys

import yaml

from .config import StrictDotMap

TASKS = {"random_cluttered_MNIST": "classification", "imagenet": "classification",
         "ade20k_semantic_seg": "semantic_seg", "coco_stuff": "semantic_seg"}


def _leaves(d, prefix=()):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _leaves(v, prefix + (k,))
        else:
            yield prefix + (k,)


def _get(d, path):
    for k in path:
        d = d[k]
    return d


def _set(d, path, value):
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value


def _merge(base, update, where="run config"):
    for k, v in update.items():
        if k not in base:
            raise KeyError(f"unknown key '{k}' in {where}")
        if isinstance(v, dict):
            _merge(base[k], v, f"{where} section '{k}'")
        else:
            base[k] = v


def _resolve(name, leaves):
    parts = tuple(name.split("."))
    matches = [p for p in leaves if p[-len(parts):] == parts]
    if len(matches) != 1:
        found = ", ".join(".".join(m) for m in matches) or "none"
        raise KeyError(f"--{name} must name exactly one config key (matches: {found})")
    return matches[0]


def _parse_value(text, default):
    value = yaml.safe_load(text)
    if isinstance(default, float) and isinstance(value, (int, str)) and not isinstance(value, bool):
        value = float(value)   # also '1e-4', which YAML reads as a string
    return value


def parse_flags(argv):
    """['--a', '1', '--flag', '--b', 'x'] -> {'a': '1', 'flag': 'true', 'b': 'x'}"""
    flags, i = {}, 0
    while i < len(argv):
        if not argv[i].startswith("--"):
            raise ValueError(f"unexpected argument {argv[i]!r}")
        name = argv[i][2:]
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            flags[name], i = argv[i + 1], i + 2
        else:
            flags[name], i = "true", i + 1
    return flags


def get_args_and_cfg(default_path, argv=None):
    flags = parse_flags(sys.argv[1:] if argv is None else argv)
    with open(default_path) as f:
        config = yaml.safe_load(f)
    run_config = flags.pop("config_path", None)
    if run_config:
        with open(run_config) as f:
            _merge(config, yaml.safe_load(f) or {})
    leaves = list(_leaves(config))
    given = set()
    for name, text in flags.items():
        path = _resolve(name, leaves)
        _set(config, path, _parse_value(text, _get(config, path)))
        given.add(path[-1])
    if config["training"]["dataset_name"] not in TASKS:
        raise ValueError(f"dataset_name must be one of {sorted(TASKS)}")
    config["task"] = {"main_task": TASKS[config["training"]["dataset_name"]]}   # set by the dataset
    config = StrictDotMap(config)
    # evaluation uses the policy's argmax unless --policy_argmax is given explicitly
    if config.training.just_evaluate and "policy_argmax" not in given:
        config.model.policy_argmax = True
    validate_config(config)
    return config


def validate_config(config):
    m = config.model
    if m.policy_loss.enabled and m.policy != "learned" and not config.training.just_evaluate:
        raise ValueError(f"policy_loss.enabled with policy '{m.policy}': the policy gradient would credit fixations "
                         f"that are never made. Use policy: learned, or policy_loss.enabled: false.")
    if (m.policy_loss.enabled and m.policy_loss.reward == "difference" and m.num_glimpses < 2
            and not config.training.just_evaluate):
        raise ValueError("the difference reward needs num_glimpses > 1")
    if m.passive_input_size > 0 and (m.policy_loss.enabled or config.utils.save_scanpaths
                                     or config.utils.replay_scanpaths):
        raise ValueError("the passive baseline has no policy: set policy_loss.enabled: false, no scanpaths")
    if config.training.resume_training and not config.training.pretrained_model_path:
        raise ValueError("resume_training needs pretrained_model_path (the run's last.pth)")
