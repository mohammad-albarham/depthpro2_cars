"""Record the Rerun viewer window (all panels) as an MP4, frame by frame, without screen capture.

A headless Rerun viewer opens the recording, the timeline is stepped one frame at a time, and each
step is screenshotted (`rerun.experimental.ViewerClient`, Rerun >= 0.43). Needs the GPU, so run it
outside any sandbox. The layout (blueprint) comes from the .rrd itself.

  python record_rerun_video.py out/viz/scene-0796_depthpro2-h_x0.5.rrd
  -> out/video/scene-0796_depthpro2-h_x0.5_rerun.mp4   (1920x1080, 12 fps = real time)
"""

from __future__ import annotations

import argparse
import io
import os
import stat
import sys
import tempfile
import time
from pathlib import Path

import av
from PIL import Image
from rerun.experimental import ViewerClient


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rrd", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: out/video/<rrd name>_rerun.mp4")
    ap.add_argument("--size", default="1920x1080", help="viewer window size in points")
    ap.add_argument("--fps", type=float, default=12.0, help="nuScenes CAM_FRONT is ~12 Hz")
    ap.add_argument("--timeline", default="frame")
    ap.add_argument("--settle", type=float, default=0.15, help="seconds to wait after each seek")
    ap.add_argument("--port", type=int, default=9877, help="keep 9876 free for your own viewer")
    args = ap.parse_args()
    out = args.out or Path("out/video") / f"{args.rrd.stem}_rerun.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    # spawn() has no window-size option, so wrap this venv's viewer with --window-size
    viewer = Path(sys.executable).with_name("rerun")
    wrapper = Path(tempfile.mkdtemp()) / "rerun_sized.sh"
    wrapper.write_text(f'#!/bin/sh\nexec "{viewer}" --window-size {args.size} "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)

    client = ViewerClient.spawn(
        headless=True, port=args.port, hide_welcome_screen=True, executable_path=str(wrapper)
    )
    try:
        client.open(str(args.rrd.resolve()))
        timeline = None
        for _ in range(60):  # wait until the recording and its timeline are loaded
            state = client.get_state()
            for rec in state.recordings:
                timeline = next((t for t in rec.timelines if t.name == args.timeline), None)
            if timeline is not None:
                break
            time.sleep(1)
        if timeline is None:
            raise SystemExit(f"Timeline {args.timeline!r} not found in {args.rrd}")
        first, last = int(timeline.start), int(timeline.end)
        n = last - first + 1
        print(f"Recording {n} frames of {args.rrd.name} -> {out}")

        container = av.open(str(out), mode="w")
        stream = None
        t0 = time.perf_counter()
        for i, seq in enumerate(range(first, last + 1)):
            client.set_time(args.timeline, sequence=seq, play_state="paused")
            time.sleep(args.settle)  # let images decode before painting
            img = Image.open(io.BytesIO(client.take_screenshot())).convert("RGB")
            if stream is None:
                w, h = img.width // 2 * 2, img.height // 2 * 2  # H.264 needs even sizes
                stream = container.add_stream("libx264", rate=round(args.fps))
                stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
                stream.options = {"crf": "18", "preset": "slow", "movflags": "+faststart"}
            if img.size != (stream.width, stream.height):
                img = img.crop((0, 0, stream.width, stream.height))
            for packet in stream.encode(av.VideoFrame.from_image(img)):
                container.mux(packet)
            if i % 30 == 0 or i == n - 1:
                print(f"  [{i + 1}/{n}] {(i + 1) / (time.perf_counter() - t0):.1f} frames/s", flush=True)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB, {n / args.fps:.1f} s)")
    finally:
        client.quit()
        os.unlink(wrapper)


if __name__ == "__main__":
    main()
