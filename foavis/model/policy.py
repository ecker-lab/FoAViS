"""Fixation policies.

- learned:  f_{t+1} ~ pi_theta(. | H_t), the softmax of the policy-head logits (paper Eq. 3). Training samples;
            evaluation takes the argmax.
- random:   uniform over the image (the random-location models).
- triangle, quadrant: fixed fixation sequences (Table 3).
- saliency, entropy:  map heuristics, argmax of a score map computed from the memory features (Table 3,
            Appendix "Heuristic policies"); `ior_radius` > 0 masks a radius around earlier
            fixations (inhibition of return).
All fixations are in [-1, 1] image coordinates, (x, y) = (column, row); the first fixation is the centre.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

POLICIES = ("learned", "random", "triangle", "quadrant", "saliency", "entropy")
ENTROPY_BLOCK = 4   # the entropy map is averaged over ENTROPY_BLOCK x ENTROPY_BLOCK memory cells

# Fixed sequences after the centre fixation, one table per number of glimpses. table[t] is fixated after
# glimpse t; the last entry is never used (the episode ends).
FIXED_SEQUENCES = {
    "triangle": {   # equilateral triangle at radius 0.65 (pentagon for 6 glimpses)
        4: [(0.0, 0.65), (-0.563, -0.325), (0.563, -0.325), (0.0, 0.0)],
        6: [(0.0, 0.65), (-0.618, 0.201), (-0.382, -0.526), (0.382, -0.526), (0.618, 0.201), (0.0, 0.0)],
    },
    "quadrant": {   # the four quadrant centres (plus a centre revisit for 6 glimpses)
        5: [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.0, 0.0)],
        6: [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.0, 0.0), (0.0, 0.0)],
    },
}


def cell_centres(n, device):
    """Centres of n grid cells along one axis in [-1, 1]."""
    return (torch.arange(n, device=device, dtype=torch.float32) + 0.5) * (2.0 / n) - 1.0


class FixationSampler(nn.Module):
    """Draws the next fixation from the policy logits.

    Sampling uses the Gumbel-max trick and places the fixation uniformly within the chosen cell; argmax takes
    the cell centre. Returns the fixation (B, 2), its log-probability log pi(f_{t+1} | H_t) and the entropy
    of pi, both (B,)."""


    def forward(self, policy_logits, argmax=False):
        B, H, W = policy_logits.shape
        flat = policy_logits.view(B, -1)
        if argmax:
            action = flat.argmax(dim=-1)
        else:
            gumbel = -torch.empty_like(flat).exponential_().log()
            action = (flat + gumbel).argmax(dim=-1)
        log_probs = F.log_softmax(flat, dim=-1)
        log_pi = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)

        fixation = torch.stack([action.remainder(W), action.div(W, rounding_mode='floor')], dim=-1).float()
        fixation = fixation + 0.5 if argmax else fixation + 0.5 + torch.empty_like(fixation).uniform_(-0.5, 0.5)
        fixation[:, 0] = fixation[:, 0] * (2.0 / W) - 1.0
        fixation[:, 1] = fixation[:, 1] * (2.0 / H) - 1.0
        return fixation, log_pi, entropy


def fixed_fixation(policy, t, num_glimpses, batch_size, device):
    tables = FIXED_SEQUENCES[policy]
    if num_glimpses not in tables:
        raise ValueError(f"policy '{policy}' is defined for {sorted(tables)} glimpses, not {num_glimpses}")
    x, y = tables[num_glimpses][t]
    return torch.tensor([[x, y]], device=device).expand(batch_size, 2)


def map_fixation(score, previous, ior_radius=None):
    """Argmax cell centre of a score map (B, H, W). With ior_radius, cells within that distance of any
    earlier fixation in `previous` (list of (B, 2)) are excluded (inhibition of return)."""
    with torch.no_grad():
        score = score.float()
        B, H, W = score.shape
        ys, xs = cell_centres(H, score.device), cell_centres(W, score.device)
        if ior_radius and previous:
            prev = torch.stack(previous, dim=1)
            d2 = ((xs.view(1, 1, 1, W) - prev[:, :, 0].view(B, -1, 1, 1)) ** 2
                  + (ys.view(1, 1, H, 1) - prev[:, :, 1].view(B, -1, 1, 1)) ** 2)
            masked = (d2 < ior_radius ** 2).any(1)
            masked &= ~masked.flatten(1).all(1).view(B, 1, 1)   # never mask everything
            score = score.masked_fill(masked, float("-inf"))
        idx = score.flatten(1).argmax(1)
        return torch.stack([xs[idx % W], ys[idx // W]], dim=1)


def saliency_map(classification_head, memory):
    """Channel mean of the classification head's last feature map before pooling (bottom-up saliency)."""
    return classification_head.conv[:-1](memory).mean(1)


def entropy_map(segmentation_head, memory, block=ENTROPY_BLOCK):
    """Per-cell entropy of the class prediction, averaged over block x block cells."""
    logp = F.log_softmax(segmentation_head.conv(memory).float(), dim=1)
    ent = -(logp.exp() * logp).sum(1)
    return F.avg_pool2d(ent.unsqueeze(1), block).squeeze(1) if block > 1 else ent
