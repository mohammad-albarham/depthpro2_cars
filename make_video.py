"""Render a shareable MP4 (input on top, Depth Pro 2 below) for a nuScenes scene.

  python make_video.py --scene scene-0796 --model depthpro2-h --scale 0.5
  -> out/video/scene-0796_depthpro2-h.mp4   (H.264, yuv420p, 12 fps = real time)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from depth_pro_2 import create_model_and_transforms
from run_nuscenes import NuScenes, colorize_inverse_depth

MODEL_LABEL = {"depthpro2-s": "S", "depthpro2-b": "B", "depthpro2-l": "L", "depthpro2-h": "H+"}


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def label(img: Image.Image, text: str, f: ImageFont.FreeTypeFont) -> None:
    """Rounded dark pill in the top-left corner, like the official demo."""
    d = ImageDraw.Draw(img)
    x0, y0, pad = 20, 18, 12
    l, t, r, b = d.textbbox((x0 + pad, y0 + pad // 2), text, font=f)
    d.rounded_rectangle((x0, y0, r + pad, b + pad // 2 + 4), radius=(b - t) // 2 + 8, fill=(15, 15, 20))
    d.text((x0 + pad, y0 + pad // 2), text, font=f, fill=(255, 255, 255))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    ap.add_argument("--scene", default="scene-0796")
    ap.add_argument("--model", default="depthpro2-h")
    ap.add_argument("--scale", type=float, default=0.5, help="model input scale (0.5 = 800x450)")
    ap.add_argument("--width", type=int, default=1280, help="output panel width (16:9 panels)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("out/video"))
    args = ap.parse_args()

    nusc = NuScenes(args.dataroot)
    scene = nusc.scene_by_name(args.scene)
    cams = nusc.channel_sequence(scene, "CAM_FRONT")
    if args.max_frames:
        cams = cams[: args.max_frames]
    K = np.array(nusc.calib[cams[0]["calibrated_sensor_token"]]["camera_intrinsic"])
    w0, h0 = cams[0]["width"], cams[0]["height"]
    W, H = round(w0 * args.scale), round(h0 * args.scale)
    PW, PH = args.width, round(args.width * h0 / w0) // 2 * 2  # panel size (even for H.264)
    footer = 56
    OW, OH = PW, 2 * PH + footer

    model, transform = create_model_and_transforms(args.model)
    print(f"{args.model} on {next(model.parameters()).device}: {len(cams)} frames -> {OW}x{OH} mp4")

    def frames():
        for sd in cams:
            img = Image.open(nusc.root / sd["filename"]).convert("RGB")
            yield transform(np.asarray(img.resize((W, H), Image.BICUBIC)))

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.scene}_{args.model}.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=12)  # nuScenes CAM_FRONT is ~12 Hz
    stream.width, stream.height, stream.pix_fmt = OW, OH, "yuv420p"
    stream.options = {"crf": "18", "preset": "slow", "movflags": "+faststart"}

    f_label, f_small = font(26), font(20)
    credit = (
        f"Apple Depth Pro 2 ({MODEL_LABEL.get(args.model, args.model)}) - metric depth from one camera"
        f"   |   nuScenes {args.scene} (CC BY-NC-SA 4.0)"
    )
    inv_scale, t0 = None, time.perf_counter()
    preds = model.infer_video(frames(), f_px=float(K[0, 0]) * args.scale)
    for i, (sd, pred) in enumerate(zip(cams, preds)):
        depth = pred["depth"].float().cpu().numpy()
        if inv_scale is None:  # fix colors on the first frame so they don't flicker
            fin = depth[np.isfinite(depth) & (depth > 0)]
            inv_scale = float(np.percentile(1.0 / fin, 99))
        rgb = Image.open(nusc.root / sd["filename"]).convert("RGB").resize((PW, PH), Image.LANCZOS)
        dep = Image.fromarray(colorize_inverse_depth(depth, inv_scale)).resize((PW, PH), Image.BICUBIC)
        label(rgb, "Input", f_label)
        label(dep, "Depth Pro 2", f_label)

        canvas = Image.new("RGB", (OW, OH), (15, 15, 20))
        canvas.paste(rgb, (0, 0))
        canvas.paste(dep, (0, PH))
        ImageDraw.Draw(canvas).text(
            (OW // 2, 2 * PH + footer // 2), credit, font=f_small, fill=(200, 200, 205), anchor="mm"
        )
        for packet in stream.encode(av.VideoFrame.from_image(canvas)):
            container.mux(packet)
        if i % 30 == 0:
            print(f"[{i + 1}/{len(cams)}] {(i + 1) / (time.perf_counter() - t0):.2f} fps")

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB, {len(cams) / 12:.1f} s)")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
