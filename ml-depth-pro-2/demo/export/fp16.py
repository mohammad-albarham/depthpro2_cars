#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import nn

from depth_pro_2 import DepthPro2

from .stream import CausalStep

FP16_TARGET_PEAK = 2.0**14


def fitting_divisor(peak: float, target: float) -> float:
    """The smallest power of two, at least 1, that divides ``peak`` down to ``target``."""
    return 2.0 ** max(0, math.ceil(math.log2(peak / target)))


def fit_fp16(model: DepthPro2, frames: torch.Tensor, target: float = FP16_TARGET_PEAK) -> float:
    """Divide the encoder's residual stream to fit its peak on ``frames``; return the divisor."""
    step = CausalStep(model, 1, *frames.shape[-2:])
    peak = residual_peak(model.encoder, lambda: [step(frame[None]) for frame in frames])
    divisor = fitting_divisor(peak, target)
    divide_residual_stream(model, divisor)
    return divisor


def fit_attention_fp16(
    model: DepthPro2, frames: torch.Tensor, window: int, target: float = FP16_TARGET_PEAK
) -> dict[int, float]:
    """The logit divisor of each block whose attention a ``window``-frame step computes unfused."""
    step = LogitPeaks(model, 1, *frames.shape[-2:])
    with torch.no_grad():
        for frame in frames:
            step(frame[None])
    divisors = {
        idx: fitting_divisor(peak, target) for idx, peak in step.peaks.items() if peak > target
    }
    if window > 1:
        masked_blocks = model.encoder.cross_frame_blocks
        divisors |= {idx: 1.0 for idx in masked_blocks if idx not in divisors}
    return dict(sorted(divisors.items()))


class LogitPeaks(CausalStep):
    """A step recording the largest unscaled attention logit of each block."""

    def __init__(self, *args, **kwargs):
        """Initialize the step with no peaks."""
        super().__init__(*args, **kwargs)
        self.peaks: dict[int, float] = {}

    def _attend(self, idx, query, key, value, mask=None):
        peak = float((query @ key.transpose(-1, -2)).abs().max())
        self.peaks[idx] = max(self.peaks.get(idx, 0.0), peak)
        return super()._attend(idx, query, key, value, mask)


def residual_peak(vit: nn.Module, run: Callable[[], object]) -> float:
    """The largest magnitude entering the two norms of ``vit``'s blocks while ``run`` runs."""
    peak = 0.0

    def record(stream: torch.Tensor) -> None:
        nonlocal peak
        peak = max(peak, float(stream.abs().max()))

    hooks = [
        norm.register_forward_pre_hook(lambda _, args: record(args[0]))
        for block in vit.blocks
        for norm in (block.norm1, block.norm2)
    ]
    try:
        with torch.no_grad():
            run()
    finally:
        for hook in hooks:
            hook.remove()
    return peak


def divide_residual_stream(model: DepthPro2, divisor: float) -> None:
    """Divide the encoder's residual stream by ``divisor``, leaving the inverse depth exact."""
    encoder = model.encoder
    divide_blocks(encoder, divisor)
    with torch.no_grad():
        for layer in encoder.reassemble:
            layer[0].weight *= divisor


def divide_blocks(vit: nn.Module, divisor: float) -> None:
    """Divide what writes a DINOv3 trunk's residual stream by ``divisor`` and rescale its norms."""
    writers = [vit.patch_embed.proj]
    writers += [m for block in vit.blocks for m in (block.attn.proj, block.mlp.fc2)]
    with torch.no_grad():
        for token in (vit.cls_token, vit.reg_token):
            token /= divisor
        for writer in writers:
            writer.weight /= divisor
            writer.bias /= divisor
    for block in vit.blocks:
        block.norm1.eps /= divisor**2
        block.norm2.eps /= divisor**2
