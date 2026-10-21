#!/usr/bin/env bash
# Render the island animation headlessly with the bpy pip module.
#   ./render.sh [quality] [out_dir] [start] [end]
# quality: preview (640x360) | medium (1280x720) | final (1920x1080)
set -euo pipefail
cd "$(dirname "$0")"
QUALITY="${1:-medium}"
OUT="${2:-renders/$QUALITY}"
START="${3:-1}"
END="${4:-192}"
# software OpenGL so Blender can start without a GPU / display
export LIBGL_ALWAYS_SOFTWARE=1 EGL_PLATFORM=surfaceless
# per-frame CPU time cap in seconds (0 = unlimited)
export FRAME_TIME_LIMIT="${FRAME_TIME_LIMIT:-40}"
mkdir -p "$OUT"
python3 -I build_island.py --engine CYCLES --quality "$QUALITY" --animation \
    --start "$START" --end "$END" --out "$OUT" 2>&1 \
  | grep -E "Saved|Error|Traceback|line [0-9]" || true
