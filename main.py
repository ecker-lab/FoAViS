"""Train or evaluate FoAViS.

    python main.py --config_path configs/paper/<model>.yaml --data_dir <data_dir> [--key value ...]
    torchrun --nproc_per_node N main.py ...          (batch_size is per GPU)
    python main.py --config_path ... --just_evaluate true --pretrained_model_path <ckpt>
"""
import random

import torch
import torch.optim as optim

from foavis.core.trainer import Trainer
from foavis.data.loaders import setup_dataloader
from foavis.io.checkpoint import CheckpointManager, load_checkpoint
from foavis.io.logger import ConsoleLogger, WandbLogger
from foavis.model.foavis import build_model
from foavis.utils.config import create_folder_name, flatten
from foavis.utils.distributed import cleanup_distributed, seed_everything, setup_distributed, wrap_model_ddp
from foavis.utils.parser import get_args_and_cfg


def freeze_policy_path(model, config, verbose):
    """Random-location models: nothing trains the policy head, so it is excluded from the optimizer (it stays
    at its initialisation; DDP also requires every tracked parameter to receive a gradient).
    train_policy_head_only: the reverse, only the policy head is trained."""
    head_only = config.model.train_policy_head_only
    if config.model.policy_loss.enabled and not head_only:
        return
    for name, p in model.named_parameters():
        in_policy = name.startswith("policy_head.")
        p.requires_grad = in_policy if head_only else not in_policy
    if verbose:
        n = sum(p.requires_grad for p in model.parameters())
        print(f"trainable parameter tensors: {n} ({'policy head only' if head_only else 'policy head frozen'})")


def make_optimizer(model, config):
    o = config.training.optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=o.lr,
                            weight_decay=o.weight_decay, betas=tuple(o.betas), eps=o.eps)
    # constant for flat_epochs, then halved every step_size_epochs
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, milestones=[o.flat_epochs], schedulers=[
        optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=o.flat_epochs),
        optim.lr_scheduler.StepLR(optimizer, step_size=o.step_size_epochs, gamma=0.5)])
    return optimizer, scheduler


def restore_optimizer(checkpoint, optimizer, config):
    """Load the checkpoint's optimizer state but start the configured schedule (ImageNet phase 2)."""
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for pg in optimizer.param_groups:
        pg['lr'] = config.training.optimizer.lr


def resume(checkpoint, optimizer, scheduler, config):
    """Continue an interrupted run from its checkpoint: optimizer state, schedule and epoch. Returns the
    first epoch to train."""
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start = int(checkpoint['epoch']) + 1
    print(f"resuming at epoch {start} of {config.training.num_epochs}")
    return start


def main():
    dist_info = setup_distributed()
    main_process = dist_info['is_main_process']
    config = get_args_and_cfg('./configs/config_default.yaml')
    config.folder_name = create_folder_name()
    if config.utils.seed is None:   # a new seed per run, saved in config.yaml
        config.utils.seed = random.randint(0, 2 ** 31 - 1)
    seed_everything(config.utils.seed, rank=dist_info['rank'])

    train_loader, valid_loader, image_size = setup_dataloader(config, dist_info=dist_info)
    model = build_model(config, image_size)
    path = config.training.pretrained_model_path
    checkpoint = load_checkpoint(model, path) if path else None
    freeze_policy_path(model, config, main_process)
    model = model.to("cuda")

    if config.training.use_compile:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True   # fall back to eager instead of failing the run
        # compile bound forwards, not the modules, so checkpoint keys stay unprefixed
        if hasattr(model, "memory"):
            model.memory.forward = torch.compile(model.memory.forward, dynamic=True)
        model.backbone.forward = torch.compile(model.backbone.forward)
    if main_process:
        print("trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer, scheduler = make_optimizer(model, config)
    if dist_info['is_distributed']:
        model = wrap_model_ddp(model, dist_info['local_rank'])
    start_epoch = 0
    if checkpoint is not None and not config.training.just_evaluate:
        if config.training.resume_training:
            start_epoch = resume(checkpoint, optimizer, scheduler, config)
        elif config.training.restore_optimizer:
            restore_optimizer(checkpoint, optimizer, config)

    if config.utils.use_wandb and main_process:
        import wandb
        wandb.init(project=config.utils.wandb_project_name, name=config.folder_name,
                   config=flatten(config.toDict()), settings=wandb.Settings(x_disable_stats=True))
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")
        logger = WandbLogger(wandb.run)
    else:
        logger = ConsoleLogger()
    ckpt = CheckpointManager(root="./model_ckpts", dataset=config.training.dataset_name, run_name=config.folder_name,
                             config=None if config.training.just_evaluate else config) if main_process else None
    try:
        Trainer(model, train_loader, valid_loader, optimizer, scheduler, config, logger,
                checkpoint_manager=ckpt, dist_info=dist_info).train(start_epoch)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
