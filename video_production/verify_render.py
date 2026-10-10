"""QA gates for the rendered video.

Usage: python video_production/verify_render.py [video]
Checks: 1920x1080, 30 fps, 2,400 frames, 80.000 s video and audio, and that the
first 12 s are intro.mp4 (scaled only; the small "Original footage" label is masked out).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def probe(path):
    out = subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-show_entries',
                                   'stream=codec_type,width,height,r_frame_rate,nb_read_frames,duration',
                                   '-of', 'json', str(path)])
    return json.loads(out)['streams']


def frame_at(path, t, scale='1920:1080'):
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-ss', f'{t:.3f}', '-i', str(path), '-frames:v', '1',
                                   '-vf', f'scale={scale}:flags=bicubic', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    return np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3).astype(np.float32)


def psnr(a, b, mask):
    mse = np.mean((a[mask] - b[mask]) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def main():
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'output' / 'FALCON_Signal13_80s.mp4'
    streams = probe(video)
    v = next(s for s in streams if s['codec_type'] == 'video')
    a = next(s for s in streams if s['codec_type'] == 'audio')
    checks = {
        'resolution 1920x1080': (v['width'], v['height']) == (1920, 1080),
        'fps 30': v['r_frame_rate'] == '30/1',
        'frames 2400': int(v['nb_read_frames']) == 2400,
        'video 80.000 s': abs(float(v['duration']) - 80.0) < 1e-3,
        'audio 80.000 s': abs(float(a['duration']) - 80.0) < 0.03,
    }
    mask = np.ones((1080, 1920), bool)
    mask[1000:1080, 0:400] = False        # "Original footage" label
    worst = 99.0
    for t in (2.0, 5.0, 8.0, 11.0):
        src = frame_at(ROOT / 'input_video' / 'intro.mp4', t)
        out = frame_at(video, t)
        worst = min(worst, psnr(src, out, mask))
    checks[f'first 12 s = intro.mp4 (min PSNR {worst:.1f} dB >= 35)'] = worst >= 35
    for k, ok in checks.items():
        print('PASS' if ok else 'FAIL', k)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == '__main__':
    main()
