"""FoAViS: the recurrent loop of paper Sec. 3 (architecture figure).

Per step t = 1..T: foveate at f_t (1) -> backbone (2) -> backproject and update the memory H_t (3) ->
policy head proposes f_{t+1} (4, not used at t = T) and the task head predicts y_t (5). f_1 is the image centre.
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .backbone import ResUNetBackbone
from .foveation import LogPolarFoveation
from .heads import ClassificationHead, PolicyHead, SegmentationHead
from .memory import MemoryModule
from .policy import POLICIES, FIXED_SEQUENCES, FixationSampler, entropy_map, fixed_fixation, map_fixation, saliency_map


@dataclass
class Outputs:
    fixations: Optional[torch.Tensor] = None     # (B, T+1, 2); the last entry is never fixated
    policy_logits: Optional[torch.Tensor] = None  # (B, T, G, G)
    class_logits: Optional[torch.Tensor] = None  # (B, T, classes), classification
    seg_logits: Optional[torch.Tensor] = None    # (B, 1, classes, S, S), the last step
    seg_logits_early: Optional[torch.Tensor] = None  # (B, T-1, classes, S, S), training only: steps 1..T-1
    seg_predictions: Optional[torch.Tensor] = None   # (B, T, S, S), evaluation only: per-step class maps
    log_pis: Optional[torch.Tensor] = None       # (B, T) log pi(f_{t+1} | H_t), when the policy is trained
    entropies: Optional[torch.Tensor] = None     # (B, T) entropy of pi(. | H_t)


class FoAViS(nn.Module):
    def __init__(self, task, num_classes, image_size, num_glimpses, policy="learned", train_policy=True,
                 stop_grad=True, policy_argmax=False, ior_radius=0.0, foveation=None, backbone=None, memory=None,
                 policy_head=None, image_channels=3):
        super().__init__()
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
        self.task, self.num_glimpses, self.policy = task, num_glimpses, policy
        self.train_policy, self.stop_grad, self.policy_argmax = train_policy, stop_grad, policy_argmax
        self.ior_radius = ior_radius if ior_radius > 0 else None

        self.foveation = LogPolarFoveation(**(foveation or {}))
        self.backbone = ResUNetBackbone(image_channels, **(backbone or {}))
        memory = dict(memory or {})
        self.memory = MemoryModule(self.backbone.out_channels, **memory)
        self.grid_size = self.memory.grid_size
        self.sampler = FixationSampler()
        self.policy_head = PolicyHead(self.memory.channels, **(policy_head or {}))
        self.classification_head = self.segmentation_head = None
        if task == "classification":
            self.classification_head = ClassificationHead(self.memory.channels, num_classes)
        else:
            self.segmentation_head = SegmentationHead(self.memory.channels, num_classes, image_size)

    def next_fixation(self, t, sampled, memory, previous, batch_size, device):
        if self.policy == "learned":
            return sampled
        if self.policy == "random":
            return torch.empty(batch_size, 2, device=device).uniform_(-1.0, 1.0)
        if self.policy in FIXED_SEQUENCES:
            return fixed_fixation(self.policy, t, self.num_glimpses, batch_size, device)
        if self.policy == "saliency":
            if self.classification_head is None:
                raise ValueError("the saliency policy needs a classification model")
            score = saliency_map(self.classification_head, memory)
        else:
            if self.segmentation_head is None:
                raise ValueError("the entropy policy needs a segmentation model")
            score = entropy_map(self.segmentation_head, memory)
        return map_fixation(score, previous, self.ior_radius)

    def forward(self, images, training, fixations=None):
        """images (B, C, S, S). `fixations` (B, T+1, 2) replays a recorded scanpath instead of the policy."""
        B, device = images.shape[0], images.device
        preset = fixations
        fixations = list(preset.to(device).unbind(1)) if preset is not None else [torch.zeros((B, 2), device=device)]
        f = fixations[0]
        h, c = self.memory.init_state(B, device)
        memories, policy_logits, class_logits, log_pis, entropies = [], [], [], [], []

        for t in range(self.num_glimpses):
            glimpse = self.foveation.glimpse(images, f)
            features = self.foveation.backproject(self.backbone(glimpse), self.grid_size)
            h, c = self.memory(features, (h, c))
            memories.append(h)
            logits = self.policy_head(h.detach() if self.stop_grad else h)
            policy_logits.append(logits)
            if self.classification_head is not None:
                class_logits.append(self.classification_head(h))

            sampled = None
            if self.train_policy or self.policy == "learned":
                sampled, log_pi, entropy = self.sampler(logits, argmax=self.policy_argmax and not training)
                if self.train_policy:
                    log_pis.append(log_pi)
                    entropies.append(entropy)
            if preset is None:
                f = self.next_fixation(t, sampled, h, fixations, B, sampled.device if sampled is not None else device)
                fixations.append(f)
            elif t >= preset.shape[1] - 1:
                fixations.append(sampled if sampled is not None else f)
            else:
                f = fixations[t + 1]

        fixations = torch.stack(fixations, dim=1)
        policy_logits = torch.stack(policy_logits, dim=1)
        out = Outputs(fixations=fixations, policy_logits=policy_logits,
                      class_logits=torch.stack(class_logits, dim=1) if class_logits else None,
                      log_pis=torch.stack(log_pis, dim=1) if log_pis else None,
                      entropies=torch.stack(entropies, dim=1) if entropies else None)
        if self.segmentation_head is not None:
            head = self.segmentation_head
            out.seg_logits = head(memories[-1]).unsqueeze(1)
            if training:
                # Task loss on every step (Eq. 5). The last step is the readout and is upsampled bilinearly, the
                # earlier ones by nearest neighbour (cheaper).
                if len(memories) > 1:
                    out.seg_logits_early = torch.stack([head(m, mode="nearest") for m in memories[:-1]], dim=1)
            else:   # class maps only: full-resolution logits of every step would take T x the memory
                out.seg_predictions = torch.stack(
                    [head(m).argmax(1) for m in memories[:-1]] + [out.seg_logits[:, 0].argmax(1)], dim=1)
        return out


def build_model(config, image_size):
    """FoAViS, or the passive baseline when model.passive_input_size > 0."""
    m = config.model
    if m.passive_input_size > 0:
        from .passive import PassiveBaseline
        return PassiveBaseline(task=config.task.main_task, num_classes=m.num_classes, image_size=image_size,
                               input_size=m.passive_input_size, backbone=m.backbone.toDict(),
                               image_channels=m.image_channels)
    return FoAViS(
        task=config.task.main_task, num_classes=m.num_classes, image_size=image_size, num_glimpses=m.num_glimpses,
        policy=m.policy, train_policy=m.policy_loss.enabled, stop_grad=m.policy_loss.stop_grad,
        policy_argmax=m.policy_argmax, ior_radius=m.ior_radius,
        foveation=m.foveation.toDict(), backbone=m.backbone.toDict(), memory=m.memory.toDict(),
        policy_head=m.policy_head.toDict(), image_channels=m.image_channels)
