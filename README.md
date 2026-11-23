# depthpro2-cars

Run Apple's [Depth Pro 2](ml-depth-pro-2/) metric depth model on nuScenes front-camera driving sequences, visualize in [Rerun](https://rerun.io), and score against LiDAR.

## Setup
```bash
source .envrc.sh      # keeps caches inside the project
uv sync               # Python 3.12
```

## Data (not in repo)
Download nuScenes v1.0-mini and extract to `data/nuscenes/`.

## Usage
```bash
python run_nuscenes.py --scene scene-0061 --model depthpro2-s   # -> out/*.rrd
rerun out/scene-0061_depthpro2-s.rrd
python make_video.py --scene scene-0796 --model depthpro2-h --scale 0.5
./bench.sh depthpro2-s depthpro2-b                               # metrics vs LiDAR
```

Metrics: AbsRel, RMSE, delta<1.25, median LiDAR/pred scale ratio.

## Licenses
`ml-depth-pro-2/` is vendored from Apple (see its `LICENSE` and `LICENSE_MODEL`). nuScenes data has its own terms and is not redistributed.
