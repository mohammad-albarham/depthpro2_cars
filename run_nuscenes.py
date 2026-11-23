"""Run Depth Pro 2 on a nuScenes front-camera sequence and visualize it in Rerun.

Logs, per CAM_FRONT frame (12 Hz, keyframes + sweeps):
  * the RGB image and the predicted metric depth (shown as a 3D point cloud via the pinhole)
  * the nearest LiDAR sweep, in 3D and projected onto the image (ground truth)
  * depth metrics vs. LiDAR: AbsRel, RMSE, delta<1.25, and the median LiDAR/pred scale ratio

Usage:
  python run_nuscenes.py --scene scene-0061 --model depthpro2-s
  rerun out/scene-0061_depthpro2-s.rrd
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from bisect import bisect_left
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from PIL import Image

from depth_pro_2 import create_model_and_transforms

MAX_DEPTH_M = 120.0  # clip predictions (sky is inf) before logging


# ---------------------------------------------------------------- nuScenes helpers
def quat_to_rot(q_wxyz) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def to_matrix(rec) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(rec["rotation"])
    T[:3, 3] = rec["translation"]
    return T


def rr_quat(q_wxyz) -> rr.Quaternion:
    w, x, y, z = q_wxyz
    return rr.Quaternion(xyzw=[x, y, z, w])


class NuScenes:
    def __init__(self, root: Path, version: str = "v1.0-mini"):
        self.root = root

        def load(name):
            return {r["token"]: r for r in json.load(open(root / version / f"{name}.json"))}

        self.scene = load("scene")
        self.sample = load("sample")
        self.sample_data = load("sample_data")
        self.calib = load("calibrated_sensor")
        self.ego_pose = load("ego_pose")
        sensor = load("sensor")
        for sd in self.sample_data.values():
            sd["channel"] = sensor[self.calib[sd["calibrated_sensor_token"]]["sensor_token"]][
                "channel"
            ]

    def scene_by_name(self, name):
        for s in self.scene.values():
            if s["name"] == name:
                return s
        names = ", ".join(sorted(s["name"] for s in self.scene.values()))
        raise SystemExit(f"Unknown scene {name!r}. Choose one of: {names}")

    def channel_sequence(self, scene, channel):
        """All sample_data (keyframes and sweeps) of one channel in a scene, in time order."""
        first_sample = self.sample[scene["first_sample_token"]]
        sd = next(
            v
            for v in self.sample_data.values()
            if v["sample_token"] == first_sample["token"] and v["channel"] == channel
        )
        # Walk back to the first sweep of the scene, then forward through the whole chain.
        while sd["prev"]:
            sd = self.sample_data[sd["prev"]]
        seq = []
        while sd:
            seq.append(sd)
            sd = self.sample_data[sd["next"]] if sd["next"] else None
        return seq

    def sensor_to_global(self, sd) -> np.ndarray:
        return to_matrix(self.ego_pose[sd["ego_pose_token"]]) @ to_matrix(
            self.calib[sd["calibrated_sensor_token"]]
        )

    def load_lidar(self, sd) -> np.ndarray:
        pts = np.fromfile(self.root / sd["filename"], dtype=np.float32).reshape(-1, 5)
        return pts[:, :3]


# ---------------------------------------------------------------- metrics
def depth_metrics(pred: np.ndarray, uv: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    p = pred[uv[:, 1], uv[:, 0]]
    ok = np.isfinite(p) & (p > 0) & (gt > 1.0) & (gt < 80.0)
    p, g = p[ok], gt[ok]
    if len(g) < 50:
        return {}
    ratio = np.maximum(p / g, g / p)
    return {
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "rmse_m": float(np.sqrt(np.mean((p - g) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "scale_gt_over_pred": float(np.median(g / p)),
        "n_points": float(len(g)),
    }


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    ap.add_argument("--scene", default="scene-0061")
    ap.add_argument("--model", default="depthpro2-s", help="depthpro2-s | -b | -l | -h")
    ap.add_argument("--estimate-focal", action="store_true", help="ignore calibration, let FOV head guess")
    ap.add_argument("--scale", type=float, default=1.0, help="resize frames (1.0 = 1600x900)")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole scene (~240 frames)")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--overlap", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--no-rrd", action="store_true", help="benchmark only: don't save a .rrd")
    ap.add_argument("--spawn", action="store_true", help="open the Rerun viewer live instead of saving")
    args = ap.parse_args()

    nusc = NuScenes(args.dataroot)
    scene = nusc.scene_by_name(args.scene)
    cams = nusc.channel_sequence(scene, "CAM_FRONT")
    if args.max_frames:
        cams = cams[: args.max_frames]
    lidars = nusc.channel_sequence(scene, "LIDAR_TOP")
    lidar_ts = [l["timestamp"] for l in lidars]

    cam_calib = nusc.calib[cams[0]["calibrated_sensor_token"]]
    K = np.array(cam_calib["camera_intrinsic"], dtype=np.float64)
    K_s = K.copy()
    K_s[:2] *= args.scale
    w0, h0 = cams[0]["width"], cams[0]["height"]
    W, H = round(w0 * args.scale), round(h0 * args.scale)
    f_px = None if args.estimate_focal else float(K_s[0, 0])

    # -------- Rerun setup
    run_name = f"{args.scene}_{args.model}" + ("_estfocal" if args.estimate_focal else "")
    if args.scale != 1.0:
        run_name += f"_x{args.scale:g}"
    if args.precision != "bf16":
        run_name += f"_{args.precision}"
    rr.init("depthpro2_nuscenes", recording_id=run_name)
    if args.spawn:
        # Use the viewer from this venv: an older `rerun` on PATH can't read newer SDK data.
        viewer = Path(sys.executable).with_name("rerun")
        rr.spawn(executable_path=str(viewer) if viewer.exists() else None)
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        rrd = Path(os.devnull) if args.no_rrd else args.out / f"{run_name}.rrd"
        rr.save(rrd)

    cam_path = "world/ego/cam_front"
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="3D: pred depth + LiDAR", origin="world"),
            rrb.Vertical(
                rrb.Spatial2DView(
                    name="RGB + LiDAR GT",
                    origin=cam_path,
                    contents=[f"{cam_path}/rgb", f"{cam_path}/lidar_gt"],
                ),
                rrb.Spatial2DView(
                    name="Depth Pro 2", origin=cam_path, contents=[f"{cam_path}/pred_color"]
                ),
                rrb.TimeSeriesView(name="metrics vs LiDAR", origin="metrics"),
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        cam_path,
        rr.Transform3D(
            translation=cam_calib["translation"], quaternion=rr_quat(cam_calib["rotation"])
        ),
        rr.Pinhole(image_from_camera=K_s, width=W, height=H, image_plane_distance=2.0),
        static=True,
    )
    rr.log(
        "description",
        rr.TextDocument(f"**{scene['name']}** — {scene['description']}", media_type="text/markdown"),
        static=True,
    )

    # -------- model
    print(f"Loading {args.model} ...")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    model, transform = create_model_and_transforms(args.model, precision=dtype)
    print(f"Device: {next(model.parameters()).device}, frames: {len(cams)} @ {W}x{H}, f_px: {f_px}")

    def frames():
        for sd in cams:
            img = Image.open(nusc.root / sd["filename"]).convert("RGB")
            if args.scale != 1.0:
                img = img.resize((W, H), Image.BICUBIC)
            yield transform(np.asarray(img))

    t0 = time.perf_counter()
    model_time, per_frame, peak_mem = 0.0, [], 0
    preds = iter(model.infer_video(frames(), window=args.window, overlap=args.overlap, f_px=f_px))
    for i, sd in enumerate(cams):
        tm = time.perf_counter()
        pred = next(preds)
        depth = pred["depth"].float().cpu().numpy()  # .cpu() syncs the GPU
        model_time += time.perf_counter() - tm
        if torch.backends.mps.is_available():
            peak_mem = max(peak_mem, torch.mps.driver_allocated_memory())
        focal = float(pred["focallength_px"])
        rr.set_time("frame", sequence=i)
        rr.set_time("time", timestamp=sd["timestamp"] * 1e-6)

        # ego pose of this camera frame
        ego = nusc.ego_pose[sd["ego_pose_token"]]
        rr.log("world/ego", rr.Transform3D(translation=ego["translation"], quaternion=rr_quat(ego["rotation"])))

        # RGB (log original JPEG bytes, cheap) and predicted depth
        if args.scale == 1.0:
            rr.log(f"{cam_path}/rgb", rr.EncodedImage(path=nusc.root / sd["filename"]))
        else:
            img = Image.open(nusc.root / sd["filename"]).convert("RGB").resize((W, H))
            rr.log(f"{cam_path}/rgb", rr.Image(np.asarray(img)).compress(jpeg_quality=90))
        d = np.where(np.isfinite(depth) & (depth < MAX_DEPTH_M), depth, 0.0)
        if i == 0:
            fin = depth[np.isfinite(depth) & (depth > 0)]
            inv_scale = float(np.percentile(1.0 / fin, 99))  # nearest ~1% of pixels = darkest red
        rr.log(
            f"{cam_path}/pred_color",
            rr.Image(colorize_inverse_depth(depth, inv_scale)).compress(jpeg_quality=90),
        )
        # float16 meters: meter=1 so depth_range is unambiguous (0..80 m), 0 = invalid
        rr.log(
            f"{cam_path}/pred_depth",
            rr.DepthImage(d.astype(np.float16), meter=1.0, colormap="turbo", depth_range=[0.0, 80.0]),
        )

        # nearest LiDAR sweep -> global -> camera
        j = min(bisect_left(lidar_ts, sd["timestamp"]), len(lidars) - 1)
        if j > 0 and abs(lidar_ts[j - 1] - sd["timestamp"]) < abs(lidar_ts[j] - sd["timestamp"]):
            j -= 1
        lid = lidars[j]
        pts = nusc.load_lidar(lid)
        pts_h = np.c_[pts, np.ones(len(pts))]
        pts_g = (nusc.sensor_to_global(lid) @ pts_h.T).T
        rr.log(
            "world/lidar",
            rr.Points3D(pts_g[:, :3], colors=[200, 200, 200], radii=0.04),
        )
        pts_c = (np.linalg.inv(nusc.sensor_to_global(sd)) @ pts_g.T).T[:, :3]
        front = pts_c[:, 2] > 1.0
        pts_c = pts_c[front]
        uvw = (K_s @ pts_c.T).T
        uv = uvw[:, :2] / uvw[:, 2:3]
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < W - 0.5) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 0.5)
        uv, z = uv[inside], pts_c[inside, 2]
        uv_i = np.round(uv).astype(int)
        rr.log(
            f"{cam_path}/lidar_gt",
            rr.Points2D(uv, radii=2.0, colors=turbo(np.clip(z / 80.0, 0, 1))),
        )

        m = depth_metrics(depth, uv_i, z)
        per_frame.append(m)
        # 0..1-ish error metrics in one chart; large-valued stats in another
        for k, v in m.items():
            group = "metrics" if k in ("abs_rel", "delta1", "scale_gt_over_pred") else "stats"
            rr.log(f"{group}/{k}", rr.Scalars(v))
        rr.log("stats/focal_px_pred", rr.Scalars(focal))
        rr.log("stats/focal_px_calib", rr.Scalars(float(K_s[0, 0])))

        if i % 20 == 0:
            fps = (i + 1) / (time.perf_counter() - t0)
            print(
                f"[{i + 1}/{len(cams)}] {fps:.2f} fps  focal={focal:.0f}px  "
                + "  ".join(f"{k}={v:.3f}" for k, v in m.items() if k != "n_points")
            )

    dt = time.perf_counter() - t0
    print(f"Done: {len(cams)} frames in {dt:.0f}s ({len(cams) / dt:.2f} fps).")

    def agg(key, fn):
        vals = [m[key] for m in per_frame if key in m]
        return float(fn(vals)) if vals else None

    summary = {
        "scene": args.scene,
        "model": args.model,
        "estimate_focal": args.estimate_focal,
        "frames": len(cams),
        "resolution": [W, H],
        "precision": args.precision,
        "abs_rel_mean": agg("abs_rel", np.mean),
        "delta1_mean": agg("delta1", np.mean),
        "rmse_m_median": agg("rmse_m", np.median),
        "scale_gt_over_pred_median": agg("scale_gt_over_pred", np.median),
        "model_fps": len(cams) / model_time,
        "peak_gpu_mem_gb": peak_mem / 1e9,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{run_name}.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary))
    if not args.spawn and not args.no_rrd:
        print(f"Open with:  .venv/bin/rerun {rrd}")


def colorize_inverse_depth(depth: np.ndarray, scale: float) -> np.ndarray:
    """Like the official demo: color 1/depth (near = red, far = blue), sky/invalid = white."""
    import matplotlib

    valid = np.isfinite(depth) & (depth > 0) & (depth < MAX_DEPTH_M)
    inv = np.where(valid, 1.0 / np.where(valid, depth, 1.0), 0.0)
    lut = (matplotlib.colormaps["Spectral_r"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    rgb = lut[np.clip(inv / scale * 255, 0, 255).astype(np.int32)]
    rgb[~valid] = 255
    return rgb


def turbo(x: np.ndarray) -> np.ndarray:
    """Cheap turbo-like colormap, x in [0, 1] -> uint8 RGB."""
    import matplotlib

    return (matplotlib.colormaps["turbo"](x)[:, :3] * 255).astype(np.uint8)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
