"""Run the unmodified FALCON tracker on the demo clips and log every frame.

Outputs evidence/<name>_track.json with the real per-frame tracker state.
Usage: python video_production/capture_tracking.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from signal13.core.tracker import TargetTracker  # noqa: E402

EVIDENCE = Path(__file__).resolve().parent / 'evidence'


def probe(path):
    out = subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate', '-of', 'csv=p=0', str(path)])
    w, h, rate = out.decode().strip().split(',')
    num, den = rate.split('/')
    return int(w), int(h), float(num) / float(den)


def frames(path):
    w, h, _ = probe(path)
    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'rawvideo',
                             '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
    size = w * h * 3
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    proc.wait()


def red_disc_bbox(frame, pad=4):
    r, g, b = (frame[..., i].astype(int) for i in range(3))
    mask = (r > 150) & (r - g > 70) & (r - b > 70)
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad
    return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)


def run(name, video, mode, roi=None, reference_image=None, diag_frames=()):
    _, _, fps = probe(video)
    tracker = TargetTracker()
    log = []
    first = None
    for i, frame in enumerate(frames(video)):
        if i == 0:
            first = frame
            if mode == 'roi':
                if roi is None:
                    roi = red_disc_bbox(frame)
                tracker.initialize(frame, roi)
                log.append({'i': 0, 't': 0.0, 'status': 'SELECTED', 'center': [roi[0] + roi[2] / 2, roi[1] + roi[3] / 2],
                            'size': [roi[2], roi[3]], 'psr': None, 'ncc': None, 'search_box': [], 'reason': 'manual ROI'})
                continue
            img = np.asarray(__import__('PIL.Image', fromlist=['Image']).open(reference_image).convert('RGB'))
            x, y, w, h = roi
            tracker.initialize_template(img[y:y + h, x:x + w])
        res = tracker.process(frame, 1.0 / fps, diagnostics=i in diag_frames)
        if i in diag_frames:
            np.savez_compressed(EVIDENCE / f'{name}_diag_f{i}.npz', **{k.replace(' ', '_'): v for k, v in res.diagnostics.items()},
                                status=res.status, psr=res.psr, ncc=res.appearance)
        log.append({
            'i': i, 't': round(i / fps, 4), 'status': res.status,
            'center': [round(float(v), 2) for v in res.center],
            'size': [int(v) for v in res.bbox_size],
            'psr': round(float(res.psr), 2), 'ncc': round(float(res.appearance), 3),
            'search_box': [round(float(v), 1) for v in res.search_box],
            'reason': res.reason,
        })
        print(name, i, res.status, f'{res.psr:.1f}', f'{res.appearance:.2f}', res.reason, flush=True)
    out = {'name': name, 'video': str(video.relative_to(ROOT)), 'fps': fps, 'mode': mode,
           'roi': list(roi), 'reference_image': str(reference_image.relative_to(ROOT)) if reference_image else None,
           'tracker': 'signal13.core.tracker.TargetTracker (default TrackerConfig)', 'frames': log}
    (EVIDENCE / f'{name}_track.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    EVIDENCE.mkdir(exist_ok=True)
    iv = ROOT / 'input_video'
    which = sys.argv[1:] or ['messi', 'frisbee']
    if 'messi' in which:
        # 32 x 32 native-pixel crop of the ball, selected from the reference still.
        run('messi', iv / 'messi-free_kick.mp4', 'reference', roi=(383, 300, 32, 32),
            reference_image=iv / 'messi_free_kick.png', diag_frames=(78,))
    if 'frisbee' in which:
        run('frisbee', iv / 'frisbee_throw.mp4', 'roi')
