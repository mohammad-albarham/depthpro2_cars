#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from depth_pro_2 import DepthPro2
from depth_pro_2.model import IMAGENET_MEAN, IMAGENET_STD, INPUT_MULTIPLE
from depth_pro_2.network.encoder import eva_finish, eva_qkv_rope

EMPTY_SLOT_LOGIT = -3.0e4
ANE_STATE_ALIGNMENT = 32
EXP_UNDERFLOW_LOGIT = -128.0
KEY_MEAN_SCALE = 2.0**-10


class CausalStep(nn.Module):
    """Unbuffered streaming step: a frame ``(1, 3, H, W)`` in [0, 1] to inverse depth."""

    def __init__(
        self,
        model: DepthPro2,
        window: int,
        height: int,
        width: int,
        logit_divisors: Mapping[int, float] | None = None,
    ):
        """Initialize an empty stream for frames of ``height`` x ``width``."""
        super().__init__()
        encoder = model.encoder
        self.model, self.window, self.height, self.width = model, window, height, width
        self.logit_divisors = dict(logit_divisors or {})
        pad_h, pad_w = -height % INPUT_MULTIPLE, -width % INPUT_MULTIPLE
        self.pad = (0, pad_w, 0, pad_h)
        self.grid = ((height + pad_h) // encoder.patch_size, (width + pad_w) // encoder.patch_size)
        self.num_patches = self.grid[0] * self.grid[1]
        self.register_buffer("rope", encoder.rope.get_embed(shape=self.grid), persistent=False)

        self.cache_names: list[str] = []
        if window > 1:
            heads = encoder.blocks[0].attn.num_heads
            dim, npt = encoder.embed_dim, encoder.num_prefix_tokens
            caches = {"filled": torch.zeros(window, ANE_STATE_ALIGNMENT)}
            for idx in sorted(encoder.cross_frame_blocks):
                kv_shape = (1, heads, window * self.num_patches, dim // heads)
                caches[f"keys_{idx}"] = torch.zeros(kv_shape)
                caches[f"values_{idx}"] = torch.zeros(kv_shape)
                caches[f"prefixes_{idx}"] = torch.zeros(window, npt, dim)
            for name, cache in caches.items():
                self.register_buffer(name, cache)
            self.cache_names = list(caches)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """Advance the stream by ``frame`` and return its inverse depth."""
        pyramid = self.model.encoder.reassemble_tokens(self.encode(frame), self.grid)
        return self.model.head(self.model.decoder(pyramid))[:, 0, : self.height, : self.width]

    def encode(self, frame: torch.Tensor) -> list[torch.Tensor]:
        """Advance the stream by ``frame`` and return its tapped patch tokens."""
        encoder = self.model.encoder
        npt = encoder.num_prefix_tokens
        patches = encoder.patch_embed(F.pad(normalize(frame), self.pad, mode="replicate"))
        joined_prefix = torch.cat([encoder.cls_token, encoder.reg_token], dim=1)
        tokens = torch.cat([joined_prefix, patches.flatten(1, 2)], dim=1)
        history = self._advance() if self.window > 1 else None
        features = []
        for idx, block in enumerate(encoder.blocks):
            if idx in encoder.cross_frame_blocks:
                tokens = self._cross_frame_attention(idx, block, tokens, history)
            else:
                attended = self._attend(idx, *eva_qkv_rope(block, tokens, self.rope, npt))
                tokens = eva_finish(block, tokens, attended)
            if idx in encoder.reassembled_blocks:
                features.append(tokens[:, npt:])
        return features

    def _attend(
        self,
        idx: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Block ``idx``'s scaled dot-product attention."""
        if idx in self.logit_divisors:
            return divided_attention(query, key, value, mask, self.logit_divisors[idx])
        return F.scaled_dot_product_attention(query, key, value, attn_mask=mask)

    def _advance(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fill one more slot; return the per-slot weights of the pooled prefix and the mask."""
        filled = self._push("filled", torch.ones_like(self.filled[:1]))[:, 0]
        empty_slots = ((1 - filled) * EMPTY_SLOT_LOGIT)[:, None].expand(-1, self.num_patches)
        prefix_rows = filled.new_zeros(self.model.encoder.num_prefix_tokens)
        mask = torch.cat([prefix_rows, empty_slots.flatten()])[None]
        return (filled / filled.sum()).view(-1, 1, 1), mask

    def _push(self, name: str, latest: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Drop the oldest slot of cache ``name``, append ``latest`` and return the cache."""
        cache = getattr(self, name)
        size = latest.shape[dim]
        updated = torch.cat([cache.narrow(dim, size, cache.shape[dim] - size), latest], dim=dim)
        cache.copy_(updated)
        return updated

    def _cross_frame_attention(
        self,
        idx: int,
        block: nn.Module,
        tokens: torch.Tensor,
        history: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        """Attend this frame's patches over the window's cached keys and values."""
        npt = self.model.encoder.num_prefix_tokens
        prefix, patches = tokens[:, :npt], tokens[:, npt:]
        query, key, value = eva_qkv_rope(block, patches, self.rope, 0)
        pooled, mask = prefix, None
        if history is not None:
            prefix_weights, mask = history
            key = self._push(f"keys_{idx}", key, dim=2)
            value = self._push(f"values_{idx}", value, dim=2)
            prefixes = self._push(f"prefixes_{idx}", prefix)
            pooled = (prefixes * prefix_weights).sum(dim=0, keepdim=True)
        _, pooled_key, pooled_value = eva_qkv_rope(block, pooled, None, npt)
        key, value = torch.cat([pooled_key, key], dim=2), torch.cat([pooled_value, value], dim=2)
        attended = self._attend(idx, query, key, value, mask)
        return torch.cat([prefix, eva_finish(block, patches, attended)], dim=1)


def divided_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
    divisor: float,
) -> torch.Tensor:
    """``F.scaled_dot_product_attention`` with its logits computed divided by ``divisor``."""
    centered_key = key - (key * KEY_MEAN_SCALE).mean(dim=-2, keepdim=True) / KEY_MEAN_SCALE
    logits = (query * (query.shape[-1] ** -0.5 / divisor)) @ centered_key.transpose(-1, -2)
    if mask is not None:
        logits = logits + mask / divisor
    # Logits minus their row max are never positive, so max=0.0 changes nothing; without it the
    # converter casts float32's largest value to float16 for the bound and warns of an overflow.
    shifted = (logits - logits.amax(dim=-1, keepdim=True)).clamp(
        min=EXP_UNDERFLOW_LOGIT / divisor, max=0.0
    )
    return (shifted * divisor).softmax(dim=-1) @ value


def normalize(image: torch.Tensor) -> torch.Tensor:
    """Normalize ``image`` ``(..., 3, H, W)`` in [0, 1] as the model's transform does."""
    mean, std = (image.new_tensor(v).view(3, 1, 1) for v in (IMAGENET_MEAN, IMAGENET_STD))
    return (image - mean) / std
