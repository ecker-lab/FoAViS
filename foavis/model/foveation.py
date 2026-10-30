"""Foveation module (paper Sec. 3.1): log-polar glimpse around the fixation, and the inverse mapping that
brings backbone features back onto the fixed memory grid in image coordinates (Sec. 3.3)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MIN_SIZE_FOR_ANTIALIASING = 256


class LogPolarFoveation(nn.Module):
    """Samples the image on an R x A grid that is uniform in angle and in log-radius around the fixation.

    The radius runs from 1 pixel to the image size S (rho_max = log S), so a fixation at the image border
    still sees the opposite side. Antialiasing: the grid is sampled at a power-of-two multiple of its size
    (at least MIN_SIZE_FOR_ANTIALIASING bins along one axis) and average-pooled down. Parameter-free.

    `glimpse` stores the fixation; `backproject` then maps features computed on that glimpse back to
    image coordinates. Fixations are in [-1, 1], (-1, -1) = top left.
    """

    def __init__(self, angular_bins=64, radius_bins=64):
        super().__init__()
        self.scale = 1
        while (angular_bins * self.scale < MIN_SIZE_FOR_ANTIALIASING
               and radius_bins * self.scale < MIN_SIZE_FOR_ANTIALIASING):
            self.scale *= 2
        self.angular_bins, self.radius_bins = angular_bins, radius_bins
        self._center = None       # (B, 2) fixation in pixels, set by glimpse()
        self._image_size = None   # (H, W)
        self.eps = 1e-8           # avoids log(0) and sqrt(0)

    def glimpse(self, image, fixation):
        """image (B, C, H, W), fixation (B, 2) in [-1, 1] -> glimpse (B, C, angular_bins, radius_bins)."""
        B, C, H, W = image.shape
        device = image.device
        assert torch.all(torch.abs(fixation) - 1e-5 <= 1), f"fixation must be in [-1, 1]: {fixation}"
        center_x = (fixation[:, 0] + 1) * (W - 1) / 2
        center_y = (fixation[:, 1] + 1) * (H - 1) / 2
        self._center = torch.stack([center_x, center_y], dim=1).to(device)
        self._image_size = (H, W)
        center_x, center_y = self._center[:, 0], self._center[:, 1]

        n_angle, n_radius = self.angular_bins * self.scale, self.radius_bins * self.scale
        theta = torch.linspace(0, 2 * math.pi, steps=n_angle, device=device)
        log_r = torch.linspace(0, torch.log(torch.tensor(float(H), device=device)), steps=n_radius, device=device)
        theta, log_r = torch.meshgrid(theta, log_r, indexing='ij')
        r = torch.exp(log_r)
        x = r.unsqueeze(0) * torch.cos(theta).unsqueeze(0) + center_x.view(B, 1, 1)
        y = -r.unsqueeze(0) * torch.sin(theta).unsqueeze(0) + center_y.view(B, 1, 1)
        grid = torch.stack((2 * x / (W - 1) - 1, 2 * y / (H - 1) - 1), dim=-1)
        glimpse = F.grid_sample(image, grid, mode='bilinear', align_corners=True)
        if self.scale > 1:
            glimpse = F.avg_pool2d(glimpse, kernel_size=self.scale, stride=self.scale)
        return glimpse

    def backproject(self, features, grid_size):
        """Log-polar features (B, C, A, R) of the last glimpse -> (B, C, grid_size, grid_size) in image space.

        Each memory cell takes the feature at its centre's log-polar coordinates (bilinear interpolation).
        """
        H, W = self._image_size
        B = features.shape[0]
        device = self._center.device
        y = torch.linspace(0, H - 1, steps=grid_size, device=device)
        x = torch.linspace(0, W - 1, steps=grid_size, device=device)
        y, x = torch.meshgrid(y, x, indexing='ij')
        dx = x.unsqueeze(0) - self._center[:, 0].view(B, 1, 1)
        dy = -(y.unsqueeze(0) - self._center[:, 1].view(B, 1, 1))
        radius = torch.sqrt(dx ** 2 + dy ** 2 + self.eps)
        theta = torch.atan2(dy, dx)
        theta = torch.where(theta < 0, theta + 2 * math.pi, theta)
        log_radius = torch.log(radius + self.eps) / torch.log(torch.tensor(float(H), device=device)) * 2 - 1
        grid = torch.stack((log_radius, (theta / (2 * math.pi)) * 2 - 1), dim=-1)
        return F.grid_sample(features, grid, mode='bilinear', align_corners=True)
