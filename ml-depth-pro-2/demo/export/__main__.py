#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import dataclasses
import math
import re
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import click
import coremltools as ct
import numpy as np
import torch
from coremltools.converters.mil.frontend.torch import register_torch_op
from PIL import Image
from torch import nn

from depth_pro_2 import PRESETS, DepthPro2, create_model_and_transforms
from depth_pro_2.model import DEFAULT_MODEL, load_checkpoint
from depth_pro_2.network.fov import FOVEstimator

from .fov import FOVStep
from .fp16 import FP16_TARGET_PEAK, fit_attention_fp16, fit_fp16
from .stream import CausalStep

REPO = Path(__file__).resolve().parents[2]
EXAMPLE_IMAGE = REPO / "assets" / "example.jpg"
FIRST_TARGET_WITH_STATE = ct.target.iOS18
LICENSE = "Apple Machine Learning Research Model license; see LICENSE_MODEL."
FIT_FRAMES = 2
PAN_PIXELS_PER_FRAME = 4
STREAM_SIZE = "480x854"
FOV_SIZE = (512, 896)  # FOVEstimator.input_size of a 16:9 frame, the demo cameras' format
SIZE_KEYS = ("demo.height", "demo.width")


@click.group()
def main():
    """Export Core ML packages for the Mac demo, macOS 15 or later."""


@main.command()
@click.option("-m", "--model", "preset", type=click.Choice(sorted(PRESETS)), required=True)
@click.option(
    "-s",
    "--size",
    "sizes",
    multiple=True,
    default=(STREAM_SIZE,),
    show_default=True,
    help="Input HxW. Repeat it for one package with a function per size, named res_<W>x<H>; the "
    "first is the default.",
)
@click.option(
    "-w",
    "--window",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Frames attended, size of the temporal window. Set to 1 to export to an image model.",
)
@click.option(
    "--fp16-peak",
    type=click.FloatRange(min=1.0, max=float(torch.finfo(torch.float16).max)),
    default=FP16_TARGET_PEAK,
    show_default=True,
    help="Largest residual-stream magnitude and unscaled attention logit to allow in float16; "
    "each is divided down to it by a power of two.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("."),
    show_default=True,
    help="Directory for the .mlpackage.",
)
def stream(
    preset: str,
    sizes: tuple[str, ...],
    window: int,
    fp16_peak: float,
    output: Path,
):
    """Export the unbuffered streaming step of Depth Pro 2."""
    dims = parse_sizes(sizes)
    if len(dims) == 1:
        height, width = dims[0]
        export_stream(load_model(preset), preset, height, width, window, output, fp16_peak)
    else:
        export_multires(lambda: load_model(preset), preset, dims, output, window, fp16_peak)


@main.command()
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("."),
    show_default=True,
    help="Directory for the .mlpackage.",
)
def fov(output: Path):
    """Export the field-of-view estimator, for the Mac demo's cameras without a focal length."""
    network = FOVEstimator()
    network.load_state_dict(load_checkpoint(PRESETS[DEFAULT_MODEL].fov_checkpoint_uri))
    export_fov(network.eval(), output)


def parse_sizes(sizes: tuple[str, ...]) -> list[tuple[int, int]]:
    """``HxW`` strings as (height, width) pairs.

    Raises:
        click.BadParameter: If a size is not ``HxW`` of positive integers or is given twice.
    """
    dims = []
    for size in sizes:
        match = re.fullmatch(r"([1-9]\d*)x([1-9]\d*)", size)
        if match is None:
            raise click.BadParameter(
                f"{size!r} is not HxW of positive integers.", param_hint="--size"
            )
        dims.append((int(match[1]), int(match[2])))
    if len(set(dims)) < len(dims):
        raise click.BadParameter("each size can be given once.", param_hint="--size")
    return dims


def load_model(preset: str) -> DepthPro2:
    """The float32 model of ``preset`` on the CPU, without the field-of-view estimator."""
    config = dataclasses.replace(PRESETS[preset], fov_checkpoint_uri=None)
    return create_model_and_transforms(config, "cpu", torch.float32)[0]


def export_stream(
    model: DepthPro2,
    preset: str,
    height: int,
    width: int,
    window: int,
    output: Path,
    fp16_peak: float = FP16_TARGET_PEAK,
) -> Path:
    """Export the stream step of ``model``, which the float16 fit rescales in place."""
    name = f"{preset}_{height}x{width}_w{window}"
    frames = panning_frames(FIT_FRAMES, height, width)
    residual_divisor = fit_fp16(model, frames, fp16_peak)
    logit_divisors = fit_attention_fp16(model, frames, window, fp16_peak)
    metadata = {
        "demo.preset": preset,
        "demo.height": height,
        "demo.width": width,
        "demo.window": window,
        "depthpro2.fp16_residual_divisor": residual_divisor,
        "depthpro2.fp16_logit_divisors": ",".join(f"{k}:{v:g}" for k, v in logit_divisors.items()),
    }

    step = CausalStep(model, window, height, width, logit_divisors)
    package = convert_step(step, frames[:1])
    package.short_description = (
        f"Depth Pro 2 {preset} unbuffered streaming step, {height}x{width}, window {window}"
    )
    if window > 1:
        package.short_description += (
            "; the earlier frames stay in the model state, so make a new state for a new video"
        )
    package.input_description["frame"] = (
        f"RGB frame (1, 3, {height}, {width}) in [0, 1], "
        f"e.g. a camera frame scaled to cover {height}x{width} and center-cropped"
    )
    package.output_description["inverse_depth"] = (
        f"Inverse depth (1, {height}, {width}) normalized by the focal length, 0 where infinitely "
        f"far; meters = f_px / ({width} * inverse_depth), with f_px the focal length in pixels "
        f"of the {height}x{width} frame"
    )
    annotate(package, metadata)
    path = output / f"{name}.mlpackage"
    save(package, path)
    return path


def export_multires(
    make_model: Callable[[], DepthPro2],
    preset: str,
    sizes: list[tuple[int, int]],
    output: Path,
    window: int,
    fp16_peak: float,
) -> None:
    """Export one package, ``<preset>_w<window>``, with the stream step at each of ``sizes``.

    The functions are named ``res_<width>x<height>`` and the first size is the default, which is
    what a caller that names no function gets. The float16 fit rescales a model in place, so each
    size starts from a new one. Weights the functions share are stored once.
    """
    names = [f"res_{width}x{height}" for height, width in sizes]
    output.mkdir(parents=True, exist_ok=True)
    # Beside the result: the packages of all sizes together are several times its size.
    with tempfile.TemporaryDirectory(dir=output) as scratch:
        paths = [
            export_stream(make_model(), preset, height, width, window, Path(scratch), fp16_peak)
            for height, width in sizes
        ]
        merge_functions(
            paths,
            names,
            output / f"{preset}_w{window}.mlpackage",
            f"Depth Pro 2 {preset} unbuffered streaming step, window {window}, one function per "
            "size: " + ", ".join(names) + "; the earlier frames stay in the model state, so make "
            "a new state for a new video",
        )


def merge_functions(
    paths: list[Path], names: list[str], destination: Path, description: str
) -> None:
    """Write the single-function packages ``paths`` as one package with a function per name.

    The first is the default, which a caller that names no function gets. A metadata entry the
    packages share is kept; one that differs, such as a size's float16 fit, becomes one entry per
    function, ``<key>.<name>``. The size is not kept: each function's input has it.
    """
    metadata_per_function = [
        dict(ct.models.MLModel(str(path), skip_model_load=True).user_defined_metadata)
        for path in paths
    ]
    metadata = {}
    for key in metadata_per_function[0]:
        if key in SIZE_KEYS:
            continue
        values = [function_metadata.get(key) for function_metadata in metadata_per_function]
        if all(value == values[0] for value in values):
            metadata[key] = values[0]
        else:
            metadata.update({f"{key}.{name}": value for name, value in zip(names, values)})
    metadata["depthpro2.functions"] = ",".join(names)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        descriptor = ct.utils.MultiFunctionDescriptor()
        for path, name in zip(paths, names):
            descriptor.add_function(str(path), "main", name)
        descriptor.default_function_name = names[0]
        merged = Path(scratch) / "merged.mlpackage"
        ct.utils.save_multifunction(descriptor, str(merged))
        package = ct.models.MLModel(str(merged), skip_model_load=True)
        package.short_description = description
        annotate(package, metadata)
        save(package, destination)


def export_fov(fov: FOVEstimator, output: Path) -> None:
    """Export ``fov`` for frames already resized to ``FOV_SIZE``."""
    height, width = FOV_SIZE
    frame = cover_crop(EXAMPLE_IMAGE, height, width)[None].contiguous()
    package = ct.convert(
        exported_program(FOVStep(fov), frame),
        inputs=[ct.TensorType(name="frame", shape=frame.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="fov", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32,
        skip_model_load=True,
    )
    package.short_description = f"Depth Pro 2 field-of-view estimator, {height}x{width}"
    package.input_description["frame"] = (
        f"RGB frame (1, 3, {height}, {width}) in [0, 1]: a frame's center 16:9 crop, "
        "resized with antialiased bilinear"
    )
    package.output_description["fov"] = (
        "Horizontal and vertical field of view (1, 2) in degrees; "
        "f_px = 0.5 * crop height / tan(vfov / 2)"
    )
    annotate(package, {})
    save(package, output / f"depthpro2-fov_{height}x{width}.mlpackage")


def convert_step(step: CausalStep, frame: torch.Tensor) -> ct.models.MLModel:
    """``step`` as a float16 ML program taking ``frame``, with its caches as state."""
    return ct.convert(
        exported_program(step, frame),
        inputs=[ct.TensorType(name="frame", shape=frame.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="inverse_depth", dtype=np.float16)],
        states=[
            ct.StateType(
                ct.TensorType(shape=getattr(step, cache).shape, dtype=np.float16), name=cache
            )
            for cache in step.cache_names
        ],
        minimum_deployment_target=FIRST_TARGET_WITH_STATE,
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
    )


@register_torch_op(override=True)
def alias(context, node) -> None:
    """Convert ``aten.alias``, which torch exports full slices as, to the identity."""
    context.add(context[node.inputs[0]], torch_name=node.name)


def exported_program(module: nn.Module, *args: torch.Tensor) -> torch.export.ExportedProgram:
    """The decomposed ``torch.export`` program of ``module``."""
    with torch.no_grad():
        return torch.export.export(module, args).run_decompositions({})


def annotate(mlmodel: ct.models.MLModel, metadata: dict[str, object]) -> None:
    """Record ``metadata``, the license and the exporting commit (marked ``-dirty`` if modified)."""
    mlmodel.license = LICENSE
    commit = subprocess.run(
        ["git", "-C", str(REPO), "describe", "--always", "--dirty", "--exclude=*"],
        capture_output=True,
        text=True,
    )
    if commit.returncode == 0:
        metadata = metadata | {"depthpro2.commit": commit.stdout.strip()}
    for key, value in metadata.items():
        mlmodel.user_defined_metadata[key] = str(value)


def save(mlmodel: ct.models.MLModel, path: Path) -> None:
    """Replace the package at ``path`` with ``mlmodel``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(path))
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    click.echo(f"wrote {path} ({size / 2**20:.1f} MiB)")


def panning_frames(count: int, height: int, width: int) -> torch.Tensor:
    """Frames ``(count, 3, H, W)`` in [0, 1] panning across the example photo."""
    pan = PAN_PIXELS_PER_FRAME
    strip = cover_crop(EXAMPLE_IMAGE, height, width + pan * (count - 1))
    return torch.stack([strip[:, :, pan * t : pan * t + width] for t in range(count)])


def cover_crop(path: Path, height: int, width: int) -> torch.Tensor:
    """The photo at ``path`` scaled and center-cropped to ``height`` x ``width``, float16-exact."""
    image = Image.open(path).convert("RGB")
    scale = max(width / image.width, height / image.height)
    size = (math.ceil(image.width * scale), math.ceil(image.height * scale))
    image = image.resize(size, Image.Resampling.LANCZOS)
    left, top = (size[0] - width) // 2, (size[1] - height) // 2
    pixels = np.array(image.crop((left, top, left + width, top + height)))
    return torch.from_numpy(pixels).permute(2, 0, 1).float().div(255).half().float()


if __name__ == "__main__":
    main()
