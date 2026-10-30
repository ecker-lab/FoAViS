"""Training objective (paper Sec. 3.6): task loss on every step (Eq. 5) plus the policy loss (Eq. 6-9)."""
import math

import torch
import torch.nn.functional as F

REWARDS = ("difference", "terminal")
REWARD_STANDARDIZATIONS = ("per_step", "abs_mean")
BASELINES = ("none", "running_mean")

# The policy loss is a sum over the T-1 decision steps (Eq. 8). It is written as a mean scaled to 3 steps,
# which is that sum at the paper's T = 4 and keeps the loss scale fixed for other T.
REFERENCE_DECISION_STEPS = 3

# Entropy bonus with a target (model.policy_loss.entropy_target > 0): its weight starts at ENTROPY_WEIGHT_INIT and is
# adapted by dual ascent in log space, lambda <- lambda * exp(ENTROPY_TARGET_LR * (target - entropy)), clipped.
ENTROPY_WEIGHT_INIT, ENTROPY_TARGET_LR, ENTROPY_WEIGHT_RANGE = 1e-3, 0.01, (1e-6, 0.1)


class LossCalculator:
    """total = L_task + L_policy. Returns (total_loss, metrics); the policy loss is only computed in training.

    model.policy_loss:
      enabled                 train the policy head (false for the random-location models)
      weight                  scale of L_policy
      reward                  difference: r_t = CE_t - CE_{t+1} (Eq. 6), terminal: -CE_T for every step
      reward_standardization  per_step (Eq. 7, over the batch at each step), abs_mean (divide by the mean |r|)
      baseline                none, running_mean (an exponential moving average of the rewards is subtracted)
      entropy_target          > 0: an entropy bonus whose weight is adapted so that the policy entropy approaches
                              this value (nats); 0: no entropy bonus
    """

    def __init__(self, config):
        self.task = config.task.main_task
        self.num_classes = config.model.num_classes
        self.ignore_index = config.model.get("seg_ignore_index", 255)
        self.cfg = config.model.policy_loss
        for key, allowed in (("reward", REWARDS), ("reward_standardization", REWARD_STANDARDIZATIONS),
                             ("baseline", BASELINES)):
            if self.cfg[key] not in allowed:
                raise ValueError(f"policy_loss.{key} must be one of {allowed}, got {self.cfg[key]!r}")
        self.training = True
        self.running_mean = None   # running-mean baseline
        self.entropy_lambda = None  # adapted entropy weight

    def __call__(self, out, targets):
        if self.task == "classification":
            return self.classification(out, targets)
        return self.segmentation(out, targets)

    # ------------------------------------------------------------------ task losses
    def classification(self, out, targets):
        logits = out.class_logits                                    # (B, T, C)
        onehot = F.one_hot(targets.to(logits.device).unsqueeze(1), num_classes=self.num_classes).float()
        B, T, C = logits.shape
        ce = F.cross_entropy(logits.reshape(B * T, C), onehot.expand(B, T, C).reshape(B * T, C),
                             reduction="none").reshape(B, T)       # (B, T)
        task_loss = ce.mean()
        total, metrics = self._add_policy_loss(task_loss, out, lambda: -ce.detach())
        with torch.no_grad():
            correct = (logits[:, -1].argmax(1) == onehot[:, -1].argmax(1)).sum().item()
            metrics["accuracy"] = correct / max(1, B)
            if not self.training:   # readout after every step (the model is causal)
                hit = (logits.argmax(-1) == onehot.argmax(-1)).float().mean(0)
                for t, v in enumerate(hit.tolist(), 1):
                    metrics[f"accuracy_g{t}"] = float(v)
        return total, metrics

    def segmentation(self, out, targets):
        seg = out.seg_logits[:, -1]                                   # (B, C, S, S) last step
        label = targets.to(seg.device)
        C = seg.shape[1]
        weight = torch.ones(C, device=seg.device)
        # down-weight the background/void class: ADE20K ignores class 0, COCO-Stuff's class 0 is kept at 0.1
        weight[self.ignore_index if self.ignore_index < C else 0] = 0.1
        ce = lambda logits, reduction="mean": F.cross_entropy(logits, label, weight=weight,
                                                              ignore_index=self.ignore_index, reduction=reduction)
        task_loss = ce(seg)
        early = out.seg_logits_early                                  # steps 1..T-1, training only
        if early is not None:
            task_loss = (sum(ce(early[:, t]) for t in range(early.shape[1])) + task_loss) / (early.shape[1] + 1)

        def rewards():
            if self.cfg.reward == "terminal" or early is None:
                return -ce(seg, "none").mean(dim=(-2, -1)).unsqueeze(1)
            steps = [early[:, t].detach() for t in range(early.shape[1])] + [seg.detach()]
            return -torch.stack([ce(s, "none").mean(dim=(-2, -1)) for s in steps], dim=1)

        total, metrics = self._add_policy_loss(task_loss, out, rewards)
        with torch.no_grad():
            preds = seg.argmax(dim=1)
            miou, pixel_acc, _, conf = compute_miou(preds, label, C, self.ignore_index)
            metrics.update(accuracy=float(miou), pixel_accuracy=float(pixel_acc), conf_matrix=conf.cpu())
            if not self.training:
                per_step = out.seg_predictions
                metrics["conf_matrix_per_glimpse"] = torch.stack(
                    [compute_miou(per_step[:, t], label, C, self.ignore_index)[3] for t in range(per_step.shape[1])]
                ).cpu()
        return total, metrics

    # ------------------------------------------------------------------ policy loss
    def _add_policy_loss(self, task_loss, out, rewards):
        metrics = {"task_loss": float(task_loss.item())}
        if not (self.training and self.cfg.enabled):
            return task_loss, metrics
        with torch.no_grad():
            r = rewards()                                             # (B, T) per-step task-loss levels -CE_t
            if self.cfg.reward == "difference":
                diff = torch.zeros_like(r)
                diff[:, :-1] = r[:, 1:] - r[:, :-1]                   # r_t = CE_t - CE_{t+1}; nothing for t = T
                r = diff
            else:
                r = r[:, -1:].contiguous()                            # -CE_T for every step
        policy_loss = self.policy_loss(out.log_pis, out.entropies, r)
        metrics["policy_loss"] = float(policy_loss.item())
        return task_loss + policy_loss, metrics

    def policy_loss(self, log_pis, entropies, r):
        """-sum_t sg[r~_t] log pi(f_{t+1} | H_t) over the T-1 decision steps, plus the entropy bonus."""
        if log_pis.shape[1] > 1:                                      # step T takes no action
            log_pis, entropies = log_pis[:, :-1], entropies[:, :-1]
            if r.shape[1] > 1:
                r = r[:, :-1]
        if self.cfg.reward_standardization == "abs_mean":
            r = r / (r.abs().mean() + 1e-8)
        r = r.expand_as(log_pis) if r.shape[1] == 1 and log_pis.shape[1] > 1 else r.view_as(log_pis)

        if self.cfg.baseline == "running_mean":
            mean = r.mean().item()
            if self.training:
                self.running_mean = mean if self.running_mean is None else 0.99 * self.running_mean + 0.01 * mean
            r = r - self.running_mean

        if self.cfg.reward_standardization == "per_step":            # Eq. 7: per step, over the batch
            r = r - r.mean(dim=0, keepdim=True)
            pooled = r.std(unbiased=False) if r.numel() > 1 else None
            if pooled is not None and float(pooled) > 1e-8:
                std = r.std(dim=0, unbiased=False, keepdim=True)
                # a step whose rewards are all equal is scaled by the pooled std instead of dividing by ~0
                r = r / torch.where(std > 1e-8 + 1e-6 * pooled, std, pooled.expand_as(std))

        loss = (-(log_pis * r)).sum(dim=1).mean() * self.cfg.weight * (REFERENCE_DECISION_STEPS / log_pis.shape[1])
        if self.cfg.entropy_target <= 0:
            return loss
        entropy = entropies.mean()
        return loss - self._entropy_weight(float(entropy.detach())) * entropy

    def _entropy_weight(self, entropy):
        if self.entropy_lambda is None:
            self.entropy_lambda = ENTROPY_WEIGHT_INIT
        if self.training and math.isfinite(entropy):
            lam = self.entropy_lambda * math.exp(ENTROPY_TARGET_LR * (self.cfg.entropy_target - entropy))
            self.entropy_lambda = min(ENTROPY_WEIGHT_RANGE[1], max(ENTROPY_WEIGHT_RANGE[0], lam))
        return self.entropy_lambda


def compute_miou(preds, targets, num_classes, ignore_index=255):
    """mIoU, pixel accuracy, per-class IoU and the confusion matrix conf[true, pred] of one batch."""
    mask = targets != ignore_index
    conf = torch.bincount(targets[mask].long() * num_classes + preds[mask].long(),
                          minlength=num_classes * num_classes).view(num_classes, num_classes)
    return (*miou_from_confusion(conf), conf)


def miou_from_confusion(conf):
    """mIoU over the classes that occur, pixel accuracy and per-class IoU from a confusion matrix."""
    tp = conf.diag()
    denom = conf.sum(0) + conf.sum(1) - tp
    valid = denom > 0
    if not valid.any():
        return 0.0, 0.0, {}
    iou = torch.zeros(conf.shape[0], device=conf.device, dtype=torch.float)
    iou[valid] = tp[valid].float() / denom[valid].float()
    total = conf.sum()
    pixel_acc = (tp.sum().float() / total).item() if total > 0 else 0.0
    return iou[valid].mean().item(), pixel_acc, {int(c): round(iou[c].item(), 4) for c in valid.nonzero(as_tuple=True)[0]}
