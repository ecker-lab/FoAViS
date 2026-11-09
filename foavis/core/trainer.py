"""Training and evaluation loop."""
import os
from datetime import datetime

import torch
import torch.distributed as dist
from torch import autocast
from tqdm import tqdm

from .losses import LossCalculator, miou_from_confusion
from ..io.scanpaths import ScanpathReader, ScanpathWriter
from ..utils.distributed import reduce_dict


class Trainer:
    def __init__(self, model, train_loader, valid_loader, optimizer, scheduler, config, logger,
                 checkpoint_manager=None, dist_info=None):
        self.model, self.config = model, config
        self.train_loader, self.valid_loader = train_loader, valid_loader
        self.optimizer, self.scheduler = optimizer, scheduler
        self.logger, self.ckpt = logger, checkpoint_manager
        self.dist_info = dist_info or {'is_distributed': False, 'is_main_process': True}
        self.is_distributed = self.dist_info['is_distributed']
        self.is_main_process = self.dist_info['is_main_process']

        self.task = config.task.main_task
        self.metric_name = "val_accuracy" if self.task == "classification" else "val_miou"
        # bf16 autocast on GPUs with native bf16 (Ampere+); semantic segmentation runs in fp32 unless enabled
        has_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
        self.use_amp = (config.training.use_amp and has_bf16 and torch.cuda.is_bf16_supported()
                        and self.task == "classification")
        self.loss = LossCalculator(config)
        self.global_step = 0
        self.epoch = 0
        self.warmup_steps = config.training.optimizer.warmup_steps
        self._base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self._step_lr = None
        self._last_ckpt_paths = []

        replay = config.utils.replay_scanpaths
        self.scanpath_reader = ScanpathReader(replay) if replay else None
        self.scanpath_writer = None
        if config.utils.save_scanpaths:
            if self.is_distributed:
                raise RuntimeError("save_scanpaths needs a single GPU")
            self.scanpath_writer = ScanpathWriter(config.utils.save_scanpaths)

    # ------------------------------------------------------------------ top level
    def train(self, start_epoch=0):
        if self.config.training.just_evaluate:
            self.evaluate()
            return
        if start_epoch > 0:   # resumed run: continue the warm-up where it stopped (if it had not ended)
            steps = len(self.train_loader)
            if self.config.utils.max_train_steps_per_epoch:
                steps = min(steps, self.config.utils.max_train_steps_per_epoch)
            self.global_step = start_epoch * steps
            self._base_lrs = [self.config.training.optimizer.lr] * len(self.optimizer.param_groups)
        best = float("-inf")   # best.pth tracks the best validation metric of this process
        for epoch in range(start_epoch, self.config.training.num_epochs):
            self.epoch = epoch
            if self.is_distributed:
                for loader in (self.train_loader, self.valid_loader):
                    if hasattr(loader.sampler, 'set_epoch'):
                        loader.sampler.set_epoch(epoch)
            metrics = self.run_epoch(self.train_loader, training=True)
            metrics['train_lr'] = self._step_lr   # rate of the epoch's last step (includes the warm-up)
            if self.is_main_process:
                print(f"epoch:{epoch} {datetime.now().strftime('%d-%m %H:%M:%S')} - Training Metrics: {metrics}")
            self.scheduler.step()   # before saving, so that a resumed run starts with the next epoch's rate

            last = epoch == self.config.training.num_epochs - 1
            if (epoch + 1) % self.config.training.validation_interval == 0 or epoch == 0 or last:
                val = self.run_epoch(self.valid_loader, training=False)
                if self.is_main_process:
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Validation Metrics: {val}")
                    value = val[self.metric_name]
                    if value > best and self.ckpt is not None:
                        best = value
                        print(f"Saved checkpoint to {self._save(epoch, self.task, value)}")
                    self._save_last(epoch, value)
                metrics.update({k: v for k, v in val.items() if not k.endswith("per_class_iou")})
            self.logger.log({**metrics, "epoch": epoch + 1})
        self.logger.close()

    def evaluate(self):
        """`training.eval_passes` validation passes; the last one is the reported result. Pass p records
        (or replays) the scanpaths of pass p."""
        for p in range(self.config.training.eval_passes):
            self.epoch = p
            val = self.run_epoch(self.valid_loader, training=False)
            if self.is_main_process:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Validation Metrics: {val}")
            self.logger.log({k: v for k, v in val.items() if not k.endswith("per_class_iou")})
        if self.scanpath_writer is not None:
            self.scanpath_writer.finalize()
        self.logger.close()

    # ------------------------------------------------------------------ one epoch
    def run_epoch(self, loader, training):
        self.model.train(training)
        self.loss.training = training
        sums, n_batches = {}, 0
        conf = conf_steps = None
        max_steps = self.config.utils.max_train_steps_per_epoch if training else 0
        test_mode = self.config.utils.use_test_mode

        for i, batch in enumerate(tqdm(loader, disable=self.config.utils.use_wandb)):
            if (test_mode and i > 100) or (max_steps and i >= max_steps):
                break
            n_batches += 1
            x = batch["image"].to("cuda", non_blocking=True)
            targets = batch["label_map"] if self.task == "semantic_seg" else batch["label"]
            image_paths = batch.get("image_path")   # absent for C-MNIST training images (generated on the fly)

            replay = None
            if self.scanpath_reader is not None and not training:
                replay = torch.from_numpy(self.scanpath_reader.get_locations(self.epoch, image_paths)).to(x.device)

            if training:
                self.optimizer.zero_grad()
            with torch.set_grad_enabled(training), autocast(device_type="cuda", enabled=self.use_amp,
                                                            dtype=torch.bfloat16 if self.use_amp else None):
                out = self.model(x, training=training, fixations=replay)
                total, metrics = self.loss(out, targets)
                if training:
                    self._step(total, test_mode)

            sums["loss"] = sums.get("loss", 0.0) + total.detach()
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    sums[k] = sums.get(k, 0.0) + v
            if "conf_matrix" in metrics:
                conf = metrics["conf_matrix"] if conf is None else conf + metrics["conf_matrix"]
            if "conf_matrix_per_glimpse" in metrics:
                cm = metrics["conf_matrix_per_glimpse"]
                conf_steps = cm if conf_steps is None else conf_steps + cm
            if self.scanpath_writer is not None and not training:
                self.scanpath_writer.write_batch(image_paths, out.fixations.detach().cpu().numpy(), self.epoch)

        return self._finalize(sums, n_batches, conf, conf_steps, training)

    def _step(self, total, test_mode):
        if test_mode:
            with torch.autograd.set_detect_anomaly(True):
                total.backward()
        else:
            total.backward()
        clip = self.config.training.clip_grad_norm
        if clip and clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip)
        self._step_lr = self.optimizer.param_groups[0]['lr']
        self.optimizer.step()
        if self.warmup_steps > 0 and self.global_step < self.warmup_steps:   # linear warm-up per step
            for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
                pg['lr'] = base_lr * (self.global_step + 1) / self.warmup_steps
        self.global_step += 1

    def _finalize(self, sums, n_batches, conf, conf_steps, training):
        loss = sums.pop("loss")
        sums["loss"] = loss.item()
        if self.is_distributed:
            sums = reduce_dict(sums, average=True)
        prefix = "train_" if training else "val_"
        out = {}
        for k, v in sums.items():
            v = round(v / n_batches, 6)
            out[prefix + k] = v * 100 if "accuracy" in k else v
        if self.task == "semantic_seg" and conf is not None:
            # mIoU from the confusion matrix accumulated over the whole set (independent of the batch size)
            conf = self._all_reduce(conf)
            miou, pix, per_class = miou_from_confusion(conf)
            out.pop(prefix + "accuracy", None)
            out[prefix + "miou"] = round(miou * 100, 6)
            out[prefix + "pixel_accuracy"] = round(pix * 100, 6)
            if not training:
                out[prefix + "per_class_iou"] = per_class
        if conf_steps is not None:
            conf_steps = self._all_reduce(conf_steps)
            for t in range(conf_steps.shape[0]):
                m, pix, _ = miou_from_confusion(conf_steps[t])
                out[f"val_miou_g{t + 1}"] = round(m * 100, 6)
                out[f"val_pixel_accuracy_g{t + 1}"] = round(pix * 100, 6)
        return out

    def _all_reduce(self, tensor):
        if not self.is_distributed:
            return tensor
        tensor = tensor.cuda()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.cpu()

    # ------------------------------------------------------------------ checkpoints and scanpaths
    def _unwrapped(self):
        return self.model.module if self.is_distributed else self.model

    def _save(self, epoch, tag, value):
        return self.ckpt.save(self._unwrapped(), epoch, tag, value, optimizer=self.optimizer,
                              scheduler=self.scheduler)

    def _save_last(self, epoch, value):
        """Rolling last-epoch checkpoints; the oldest is removed only after the new one is written."""
        if self.ckpt is None:
            return
        self._last_ckpt_paths.append(self._save(epoch, "last", value))
        while len(self._last_ckpt_paths) > max(1, self.config.training.keep_last_n_checkpoints):
            stale = self._last_ckpt_paths.pop(0)
            if os.path.exists(stale):
                os.remove(stale)
