#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import torch
from torch import nn

from depth_pro_2.model import IMAGENET_MEAN, IMAGENET_STD
from depth_pro_2.network.fov import FOVEstimator


class FOVStep(nn.Module):
    """The FoV estimator, without its resize and crop, on a frame ``(1, 3, H, W)`` in [0, 1]."""

    def __init__(self, fov: FOVEstimator):
        """Wrap ``fov``, normalizing the frame as the Python transform does."""
        super().__init__()
        self.fov = fov
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """Horizontal and vertical field of view ``(1, 2)`` in degrees of ``frame``."""
        fov = self.fov
        tokens, grid_size = fov.encode((frame - self.mean) / self.std)
        logits = fov.head(tokens, grid_size)
        return fov.min_fov_deg + torch.sigmoid(logits) * (fov.max_fov_deg - fov.min_fov_deg)
