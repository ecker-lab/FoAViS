"""Checkpoints: {'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'epoch', 'metric'}.
The released checkpoints hold model_state_dict, epoch and metric only."""
import os

import torch
import yaml


class CheckpointManager:
    """Writes <root>/<dataset>/<run name>/: config.yaml, foavis_<epoch>_<tag>_<metric>.pth and the links
    best.pth / last.pth to the newest best and last-epoch checkpoints."""

    def __init__(self, root, dataset, run_name, config=None):
        self.dir = os.path.join(root, dataset, run_name)
        if config is not None:
            os.makedirs(self.dir, exist_ok=True)
            with open(os.path.join(self.dir, "config.yaml"), "w") as f:
                yaml.dump(config.toDict(), f, default_flow_style=False, sort_keys=False)

    def save(self, model, epoch, tag, metric, optimizer=None, scheduler=None):
        os.makedirs(self.dir, exist_ok=True)
        name = f"foavis_{epoch}_{tag}_{metric:.3f}.pth"
        path = os.path.join(self.dir, name)
        checkpoint = {'model_state_dict': model.state_dict(), 'epoch': epoch, 'metric': metric}
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(checkpoint, path)
        link = os.path.join(self.dir, "last.pth" if tag == "last" else "best.pth")
        try:
            if os.path.lexists(link):
                os.remove(link)
            os.symlink(name, link)
        except OSError:   # no symlinks on this file system
            pass
        return path


def load_checkpoint(model, path):
    """Load model weights; parameters whose shape differs or that are missing keep their initialisation and
    are listed. Returns the checkpoint dict (optimizer/scheduler state for resuming)."""
    checkpoint = torch.load(path, map_location="cpu")
    pretrained = checkpoint['model_state_dict']
    own = model.state_dict()
    compatible = {k: v for k, v in pretrained.items() if k in own and v.shape == own[k].shape}
    mismatched = sorted(k for k in pretrained if k in own and k not in compatible)
    missing = sorted(set(own) - set(compatible) - set(mismatched))
    unexpected = sorted(set(pretrained) - set(own))
    print(f"Loaded {path} (epoch {checkpoint.get('epoch', '?')}, metric {checkpoint.get('metric', '?')})")
    for name, keys in (("missing in the checkpoint", missing), ("shape mismatch, kept initialisation", mismatched),
                       ("unexpected, ignored", unexpected)):
        if keys:
            print(f"Warning: {len(keys)} parameters {name}: {keys}")
    model.load_state_dict(compatible, strict=False)
    return checkpoint
