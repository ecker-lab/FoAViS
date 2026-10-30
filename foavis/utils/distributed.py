"""Multi-GPU training with torchrun (DistributedDataParallel) and seeding."""
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def seed_everything(seed, rank=0):
    """Seeds python, numpy and torch with seed + rank (augmentations differ across ranks)."""
    seed += rank
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(seed):
    """DataLoader workers: python and numpy seeded with seed + worker id (torch is seeded per worker already)."""
    def init(worker_id):
        random.seed((seed + worker_id) % 2 ** 32)
        np.random.seed((seed + worker_id) % 2 ** 32)
    return init


def setup_distributed():
    """Initialises the process group under torchrun; returns rank, local_rank, world_size, is_distributed,
    is_main_process."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return dict(rank=0, local_rank=0, world_size=1, is_distributed=False, is_main_process=True)
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return dict(rank=rank, local_rank=local_rank, world_size=int(os.environ["WORLD_SIZE"]), is_distributed=True,
                is_main_process=rank == 0)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(model, local_rank, find_unused_parameters=False):
    if not dist.is_initialized():
        return model
    model = model.to(torch.device(f"cuda:{local_rank}"))
    return DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                   find_unused_parameters=find_unused_parameters)


def reduce_dict(values, average=True):
    """Sums (or averages) a dict of numbers over all ranks."""
    if not dist.is_initialized() or dist.get_world_size() < 2:
        return values
    names = sorted(values)
    with torch.no_grad():
        t = torch.stack([torch.tensor(values[k], device="cuda") for k in names])
        dist.all_reduce(t)
        if average:
            t /= dist.get_world_size()
    return {k: v.item() for k, v in zip(names, t)}
