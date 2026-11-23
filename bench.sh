#!/bin/bash
# Usage: ./bench.sh depthpro2-s depthpro2-b ...   -> out/bench/<scene>_<model>.json
cd "$(dirname "$0")" && source .envrc.sh
SCENES="${SCENES:-scene-0061 scene-0655 scene-0796 scene-1077}"
for m in "$@"; do
  for s in $SCENES; do
    .venv/bin/python run_nuscenes.py --model "$m" --scene "$s" --no-rrd --max-frames "${MAX_FRAMES:-0}" --out "${OUT:-out/bench}" 2>&1 | grep -E "SUMMARY|Error|error" || exit 1
  done
done
