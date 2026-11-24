"""Download what the demo needs: nuScenes v1.0-mini and/or Depth Pro 2 weights.

  python download_data.py --nuscenes-mini --accept-nuscenes-terms   # ~4 GB download, ~5 GB on disk
  python download_data.py --weights h                               # H+ weights (3.5 GB) + FOV head
  python download_data.py --weights s b l h                         # all model sizes

Both are third-party data with their own licenses; nothing here is redistributed by this project.
  * nuScenes: CC BY-NC-SA 4.0 + nuScenes Terms of Use (non-commercial),
    https://www.nuscenes.org/terms-of-use . The official route is to create a free account at
    https://www.nuscenes.org/nuscenes#download and download "Mini" there; this script fetches the
    same public v1.0-mini archive only after you confirm that you accept those terms.
  * Depth Pro 2 weights: Apple Machine Learning Research Model License (research use only),
    see ml-depth-pro-2/LICENSE_MODEL. Weights also download automatically on first use; this
    command pre-fetches them and verifies their size (an interrupted download once left a
    truncated file that failed to load).

Downloads resume if interrupted and are checked against the server's Content-Length.
Run `source .envrc.sh` first so weights land in .cache/torch/hub/checkpoints/.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

NUSCENES_MINI_URL = "https://www.nuscenes.org/data/v1.0-mini.tgz"
NUSCENES_TERMS_URL = "https://www.nuscenes.org/terms-of-use"
CHUNK = 1 << 20


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def download(url: str, dest: Path, attempts: int = 5) -> Path:
    """Download url to dest; resume partial files, retry cut connections, verify the size.

    Connections are sometimes closed a few KB early (seen through proxies), so the size is
    always checked against Content-Length and the transfer resumed with an HTTP Range request.
    """
    total = remote_size(url)
    if dest.exists() and dest.stat().st_size == total:
        print(f"  ok (already complete): {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f"  downloading {url}\n    -> {dest} ({total / 1e9:.2f} GB)")
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > total:
            part.unlink()
            have = 0
        if have == total:
            break
        if have:
            print(f"    resuming at {have / 1e9:.2f} GB (attempt {attempt}/{attempts})")
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:  # server ignored the range request: start over
                    have = 0
                with open(part, "ab" if have else "wb") as f:
                    done, last_pct = have, -1
                    while chunk := r.read(CHUNK):
                        f.write(chunk)
                        done += len(chunk)
                        pct = int(100 * done / total)
                        if pct != last_pct and pct % 5 == 0:
                            print(f"    {pct:3d}%  {done / 1e9:.2f} / {total / 1e9:.2f} GB", flush=True)
                            last_pct = pct
        except OSError as e:  # dropped connection, timeout, ...: retry from where we are
            print(f"    connection error: {e}")
    size = part.stat().st_size if part.exists() else 0
    if size != total:
        raise SystemExit(f"Incomplete download ({size} of {total} bytes). Re-run to resume.")
    part.replace(dest)
    return dest


def nuscenes_mini(root: Path, accept_terms: bool, keep_archive: bool) -> None:
    if (root / "v1.0-mini" / "scene.json").exists() and (root / "sweeps" / "CAM_FRONT").exists():
        print(f"nuScenes v1.0-mini already present in {root}")
        return
    if not accept_terms:
        print(
            "nuScenes is licensed under CC BY-NC-SA 4.0 plus the nuScenes Terms of Use\n"
            f"(non-commercial use only): {NUSCENES_TERMS_URL}\n"
            "Read them, then re-run with --accept-nuscenes-terms, or create an account and\n"
            "download 'Mini' from https://www.nuscenes.org/nuscenes#download and extract it to\n"
            f"{root}/ (folders samples/, sweeps/, maps/, v1.0-mini/)."
        )
        sys.exit(1)
    archive = download(NUSCENES_MINI_URL, root.parent / "v1.0-mini.tgz")
    print(f"  extracting to {root} ...")
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter="data")  # "data" filter blocks absolute paths / path escapes
    if not keep_archive:
        archive.unlink()
    print(f"nuScenes v1.0-mini ready in {root}")


def weights(sizes: list[str]) -> None:
    import torch

    from depth_pro_2.model import PRESETS

    print(
        "Depth Pro 2 weights are licensed under the Apple Machine Learning Research Model License\n"
        "(research use only); see ml-depth-pro-2/LICENSE_MODEL."
    )
    ckpt_dir = Path(torch.hub.get_dir()) / "checkpoints"  # where the model loader looks
    urls = []
    for s in sizes:
        cfg = PRESETS[f"depthpro2-{s}"]
        urls += [cfg.checkpoint_uri, cfg.fov_checkpoint_uri]
    for url in dict.fromkeys(u for u in urls if u):  # de-duplicate, keep order
        download(url, ckpt_dir / url.rsplit("/", 1)[-1])
    print(f"Weights ready in {ckpt_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--nuscenes-mini", action="store_true", help="download + extract nuScenes v1.0-mini")
    ap.add_argument("--accept-nuscenes-terms", action="store_true", help=f"you accept {NUSCENES_TERMS_URL}")
    ap.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    ap.add_argument("--keep-archive", action="store_true", help="keep the 4 GB .tgz after extracting")
    ap.add_argument("--weights", nargs="+", choices=["s", "b", "l", "h"], help="model sizes to fetch")
    args = ap.parse_args()
    if not (args.nuscenes_mini or args.weights):
        ap.print_help()
        return
    if args.nuscenes_mini:
        nuscenes_mini(args.dataroot, args.accept_nuscenes_terms, args.keep_archive)
    if args.weights:
        weights(args.weights)


if __name__ == "__main__":
    main()
