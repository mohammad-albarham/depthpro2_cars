# depthpro2-cars

Run Apple's [Depth Pro 2](ml-depth-pro-2/) metric depth model on nuScenes front-camera driving
sequences, score it against LiDAR, visualize it in [Rerun](https://rerun.io), and render shareable
videos.

- Full write-up (method, metrics, assumptions, results): [`explination/technical_details.html`](explination/technical_details.html)
- Instructions for coding agents: [`AGENTS.md`](AGENTS.md)

## Setup

Tested on an Apple M3 Max (64 GB). The model runs on the Apple GPU (MPS).

```bash
source .envrc.sh                                   # keep all caches inside the project
/opt/homebrew/bin/python3.12 -m venv .venv         # Python 3.12
.venv/bin/pip install -e ./ml-depth-pro-2 "rerun-sdk>=0.24"
```

Use **`.venv/bin/rerun`** to open recordings. An older `rerun` elsewhere on your PATH may not read
files written by rerun-sdk 0.43 (you'd see an empty scene and a "transport error").

Model weights download automatically on first use into `.cache/torch/hub/checkpoints/`
(S 97 MB, B 386 MB, L 1.3 GB, H+ 3.5 GB).

## Data (not in repo)

Download [nuScenes](https://www.nuscenes.org/) **v1.0-mini** and extract it to `data/nuscenes/`
(folders `samples/`, `sweeps/`, `maps/`, `v1.0-mini/`).

## Usage

```bash
source .envrc.sh

# One scene → live Rerun viewer (H+ at half resolution: ~1.4 fps, same accuracy as full res)
.venv/bin/python run_nuscenes.py --model depthpro2-h --scale 0.5 --scene scene-0796 --spawn

# Or save a recording and open it later
.venv/bin/python run_nuscenes.py --model depthpro2-h --scale 0.5 --scene scene-0796 --out out/viz
.venv/bin/rerun out/viz/scene-0796_depthpro2-h_x0.5.rrd

# Shareable MP4: input on top, colored depth below
.venv/bin/python make_video.py --scene scene-0796        # → out/video/scene-0796_depthpro2-h.mp4

# Benchmark: 4 models × 4 scenes × 64 frames, full resolution (~65 min), then the ranking table
MAX_FRAMES=64 OUT=out/bench64 ./bench.sh depthpro2-s depthpro2-b depthpro2-l depthpro2-h
.venv/bin/python compare.py out/bench64
```

Main `run_nuscenes.py` options: `--model` (`depthpro2-s|b|l|h`), `--scene`, `--scale`,
`--max-frames`, `--window`/`--overlap` (16/4), `--precision` (bf16), `--estimate-focal`,
`--no-rrd`, `--spawn`.

## How scoring works (short version)

- **Ground truth:** the nearest LiDAR sweep in time, projected into the camera with ego-motion
  compensation. Only points **in front of the camera** (z > 1 m) and **inside the image** are kept.
- **Compared quantity:** depth along the camera axis (z-depth), in meters. No scale alignment.
- **Metrics** (LiDAR depth 1–80 m): **AbsRel** (mean relative error, lower is better),
  **δ<1.25** (share of points within 25%, higher is better), **RMSE** in meters, and the
  **LiDAR ÷ predicted** scale ratio (1.0 = no bias).

## Results (4 scenes × 64 frames, full resolution)

| Model | AbsRel ↓ | δ<1.25 ↑ | RMSE (m) | LiDAR ÷ pred | fps (M3 Max) |
|---|---|---|---|---|---|
| **H+** | **0.161** | **85.5%** | **5.23** | 0.889 | 0.12 |
| L | 0.179 | 83.2% | 5.28 | 0.877 | 0.23 |
| B | 0.182 | 79.9% | 5.85 | 0.888 | 0.67 |
| S | 0.207 | 72.8% | 6.38 | 0.854 | 1.47 |

- H+ is the most accurate. At half resolution it is ~11× faster with the same accuracy (tested on one scene).
- All models place objects ~10–15% too far away.
- Night scenes are the hardest case for every model.

See the [write-up](explination/technical_details.html) for per-scene results, speed experiments,
assumptions and limitations.

## Project layout

| Path | Purpose |
|---|---|
| `run_nuscenes.py` | Model + LiDAR scoring → Rerun `.rrd` and metrics `.json` |
| `make_video.py` | Model → MP4 |
| `bench.sh`, `compare.py` | Run the benchmark / print tables |
| `ml-depth-pro-2/` | Apple's code (vendored, unmodified) |
| `out/` | Results, recordings, videos (`out/bench64/` = reference benchmark) |
| `explination/` | Technical write-up and figures |

## Licenses

`ml-depth-pro-2/` is vendored from Apple: code under its `LICENSE`, weights under `LICENSE_MODEL`
(research use only). nuScenes data is CC BY-NC-SA 4.0 (non-commercial, attribution) and is not
redistributed. This project is for research use only.
