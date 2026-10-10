#!/usr/bin/env bash
# Rebuild the 80 s video end to end.  Usage: video_production/make_video.sh <work_dir>
# Needs: ffmpeg, numpy, Pillow, kokoro-onnx, soundfile, and the Kokoro v1.0 model files
# (kokoro-v1.0.onnx, voices-v1.0.bin from github.com/thewh1teagle/kokoro-onnx releases) in <work_dir>/tts.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$1"
python3 "$HERE/capture_tracking.py"
python3 "$HERE/narration.py" "$WORK/tts" "$WORK/vo"
python3 "$HERE/render_video.py" "$WORK"
python3 "$HERE/audio_mix.py" "$WORK"
mkdir -p "$HERE/output"
ffmpeg -v error -y -i "$WORK/picture.mp4" -i "$WORK/soundtrack.wav" -map 0:v -map 1:a -c:v libx264 -preset slow -crf 21 -pix_fmt yuv420p \
  -c:a aac -b:a 192k -ar 48000 -t 80 -movflags +faststart "$HERE/output/FALCON_Signal13_80s.mp4"
python3 "$HERE/verify_render.py"
