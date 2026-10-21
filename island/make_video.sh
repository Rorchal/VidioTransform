#!/usr/bin/env bash
# Assemble rendered PNG frames into an H.264 mp4 with a gentle fade in/out.
#   ./make_video.sh [frames_dir] [output.mp4] [fps]
set -euo pipefail
cd "$(dirname "$0")"
DIR="${1:-renders/medium}"
OUT="${2:-island.mp4}"
FPS="${3:-24}"
N=$(ls "$DIR"/frame_*.png | wc -l)
DUR=$(python3 -c "print($N/$FPS)")
ffmpeg -y -framerate "$FPS" -pattern_type glob -i "$DIR/frame_*.png" \
  -vf "fade=t=in:st=0:d=0.8,fade=t=out:st=$(python3 -c "print(max(0,$DUR-0.8))"):d=0.8,format=yuv420p" \
  -c:v libx264 -preset slow -crf 18 -movflags +faststart "$OUT"
echo "wrote $OUT ($N frames, ${DUR}s)"
