"""One short training epoch per model type: checks that training, validation and checkpointing run.

Needs a GPU and the data (set DATA_DIR, default ./data; C-MNIST downloads MNIST/EMNIST on first use).
Each case trains 3 steps and validates on at most 100 batches (--use_test_mode), about a minute.
"""
import glob
import os
import subprocess
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(os.environ.get("DATA_DIR", os.path.join(REPO, "data")))

CASES = {  # config -> dataset folder it needs under DATA_DIR
    "cmnist_learned": "MNIST",
    "cmnist_random": "MNIST",
    "cmnist_passive": "MNIST",
    "ade20k_learned": "ADEChallengeData2016",
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("config", sorted(CASES))
def test_one_epoch(config, tmp_path):
    if not os.path.isdir(os.path.join(DATA_DIR, CASES[config])):
        pytest.skip(f"no {CASES[config]} under {DATA_DIR}")
    # main.py reads ./configs and writes ./model_ckpts relative to the working directory
    os.symlink(os.path.join(REPO, "configs"), tmp_path / "configs")
    cmd = [sys.executable, os.path.join(REPO, "main.py"),
           "--config_path", f"configs/paper/{config}.yaml", "--data_dir", DATA_DIR,
           "--pretrained_model_path", "null", "--num_epochs", "1", "--validation_interval", "1",
           "--max_train_steps_per_epoch", "3", "--use_test_mode", "--batch_size", "4",
           "--num_workers", "2", "--use_compile", "false", "--use_wandb", "false",
           "--seed", "0"]
    r = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert glob.glob(str(tmp_path / "model_ckpts" / "*" / "*" / "*.pth")), "no checkpoint written"
