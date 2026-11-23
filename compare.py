"""Summarize bench JSONs into a table: python compare.py out/bench64"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "out/bench64")
runs = [json.loads(p.read_text()) for p in sorted(root.glob("*.json"))]
order = ["depthpro2-s", "depthpro2-b", "depthpro2-l", "depthpro2-h"]
by_model = defaultdict(list)
for r in runs:
    by_model[r["model"]].append(r)

print("\nPer scene: AbsRel (lower better) / delta1 (higher better)")
scenes = sorted({r["scene"] for r in runs})
print(f"{'model':<14}" + "".join(f"{s:>18}" for s in scenes))
for m in order:
    if m not in by_model:
        continue
    cells = {r["scene"]: f"{r['abs_rel_mean']:.3f} / {r['delta1_mean']:.2f}" for r in by_model[m]}
    print(f"{m:<14}" + "".join(f"{cells.get(s, '-'):>18}" for s in scenes))

print("\nAverage over scenes")
print(f"{'model':<14}{'AbsRel':>8}{'delta1':>8}{'RMSE m':>8}{'GT/pred':>9}{'fps':>7}{'GPU GB':>8}")
for m in order:
    rs = by_model.get(m)
    if not rs:
        continue
    avg = lambda k: np.mean([r[k] for r in rs])  # noqa: E731
    print(
        f"{m:<14}{avg('abs_rel_mean'):>8.3f}{avg('delta1_mean'):>8.3f}{avg('rmse_m_median'):>8.2f}"
        f"{avg('scale_gt_over_pred_median'):>9.3f}{avg('model_fps'):>7.2f}"
        f"{max(r['peak_gpu_mem_gb'] for r in rs):>8.1f}"
    )
