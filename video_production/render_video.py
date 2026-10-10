"""Render the 80 s FALCON / Signal 13 video (1920x1080, 30 fps, 2,400 frames).

Usage:
    python video_production/render_video.py <work_dir> [--frames a:b] [--stills f1,f2,...]

<work_dir> must contain vo/ (narration.py output). Decoded source frames are
cached there. The final picture is written to <work_dir>/picture.mp4; mix it
with the audio from audio_mix.py (see make_video.sh).
"""
import json
import math
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from gfx import *  # noqa: E402,F401,F403
import gfx  # noqa: E402

FPS = 30
TOTAL = 2400
SB = json.loads((HERE / 'storyboard.json').read_text())
SHOTS = {s['id']: s for s in SB['shots']}
TRACK = {n: json.loads((HERE / 'evidence' / f'{n}_track.json').read_text()) for n in ('messi', 'frisbee')}
with np.load(HERE / 'evidence' / 'messi_diag_f78.npz') as _z:
    DIAG = {k: _z[k] for k in _z.files}

WORK = None
SRC = {}
BGIMG = None
REF_PNG = None
CROP = None          # colour 32x32 reference crop of the ball
AUTHORS = []


# ======================================================================== setup
def decode(name, path, w, h):
    raw = WORK / f'{name}_{w}x{h}.raw'
    if not raw.exists():
        subprocess.check_call(['ffmpeg', '-v', 'error', '-y', '-i', str(path), '-f', 'rawvideo',
                               '-pix_fmt', 'rgb24', str(raw)])
    n = raw.stat().st_size // (w * h * 3)
    return np.memmap(raw, np.uint8, 'r', shape=(n, h, w, 3))


def captions_schedule():
    takes = json.loads((WORK / 'vo' / 'takes.json').read_text())['takes']
    gap = SB['vo_gap']
    rows = []
    for shot, off in SB['vo_offsets'].items():
        base = SHOTS[shot.rstrip('b')]['start'] / FPS
        t = base + off
        for p in takes[shot]:
            rows.append({'shot': shot, 'text': p['text'], 'wav': p['wav'], 'start': round(t, 3),
                         'end': round(t + p['dur'], 3)})
            t += p['dur'] + gap
    return rows


def init(work):
    global WORK, BGIMG, REF_PNG, CROP, AUTHORS, CAPS
    WORK = Path(work)
    SRC['intro'] = decode('intro', ROOT / 'input_video/intro.mp4', 1280, 720)
    SRC['messi'] = decode('messi', ROOT / 'input_video/messi-free_kick.mp4', 1280, 720)
    SRC['frisbee'] = decode('frisbee', ROOT / 'input_video/frisbee_throw.mp4', 640, 360)
    BGIMG = make_background()
    REF_PNG = np.asarray(Image.open(ROOT / 'input_video/messi_free_kick.png').convert('RGB'))
    x, y, w, h = TRACK['messi']['roi']
    CROP = REF_PNG[y:y + h, x:x + w].copy()
    AUTHORS = []
    # square, face-centred crops (x0, y0, x1, y1) chosen by eye for each portrait
    for name, sid, fn, box in (('Abu Bakar Siddique', '2305059', 'Abu Bakar Siddique.jpeg', (252, 620, 1352, 1720)),
                               ('Wahidul Hoque', '2305054', 'Wahidul Hoque.jpg', (31, 129, 691, 789))):
        im = Image.open(ROOT / 'Authors_pic' / fn).convert('RGB')
        AUTHORS.append((name, sid, im.crop(box).resize((360, 360), Image.LANCZOS)))
    CAPS = captions_schedule()


CAPS = []


# ======================================================================== shared pieces
def canvas():
    return BGIMG.copy()


def draw_captions(img, f):
    t = f / FPS
    for i, c in enumerate(CAPS):
        nxt = CAPS[i + 1]['start'] if i + 1 < len(CAPS) else 99
        shot_end = SHOTS[c['shot'].rstrip('b')]['end'] / FPS
        end = min(c['end'] + 0.45, nxt, shot_end)
        if c['shot'] == 's2':
            continue
        if c['start'] - 0.05 <= t < end:
            a = clamp((t - c['start'] + 0.05) / 0.12) * clamp((end - t) / 0.1)
            caption(img, c['text'], a)
            return


VIDEO_BOX = (40, 156, 1400, 788)          # x, y, w, h of the main media panel
SIDE_X = 1474


def media_frame(img, frame, box=VIDEO_BOX):
    x, y, w, h = box
    d = D(img)
    d.rounded_rectangle([x - 3, y - 3, x + w + 3, y + h + 3], radius=8, outline=BORDER, width=2)
    paste_fit(img, frame, box)


def to_panel(pt, src_w, box=VIDEO_BOX):
    s = box[2] / src_w
    return box[0] + pt[0] * s, box[1] + pt[1] * s, s


def side_panel(img, a=1.0):
    d = D(img)
    d.rounded_rectangle([SIDE_X, VIDEO_BOX[1] - 3, 1880, VIDEO_BOX[1] + VIDEO_BOX[3] + 3], radius=10,
                        fill=rgba(PANEL, a), outline=rgba(BORDER, a), width=2)
    return d


def status_block(img, rec, src_t, kind, a=1.0, live_tile=None, lost=False):
    d = side_panel(img, a)
    x = SIDE_X + 26
    y = VIDEO_BOX[1] + 26
    chip(d, x, y, 'ACTUAL TRACKER OUTPUT', MINT, a)
    st = rec['status']
    col = MINT if st == 'TRACKING' else AMBER
    text(d, (x, y + 66), st, font('bold', 44), col, alpha=a)
    sub = 'Pattern match accepted' if st == 'TRACKING' else ('Match lost - searching' if lost else 'No trusted match')
    text(d, (x, y + 124), sub, font('reg', 22), MUTED, alpha=a)
    psr = rec['psr'] if rec['psr'] is not None else 0
    ncc = rec['ncc'] if rec['ncc'] is not None else 0
    text(d, (x, y + 184), 'PSR', font('mono', 17), MUTED, alpha=a)
    text(d, (x + 200, y + 184), 'NCC', font('mono', 17), MUTED, alpha=a)
    text(d, (x, y + 208), f'{psr:5.1f}'.strip(), font('monob', 52), TEXT, alpha=a)
    text(d, (x + 200, y + 208), f'{ncc:5.2f}'.strip(), font('monob', 52), TEXT, alpha=a)
    text(d, (x, y + 278), 'Match scores, not accuracy.', font('reg', 18), MUTED, alpha=a)
    # live working-template tile
    if live_tile is not None:
        ty = y + 330
        tile = Image.fromarray(live_tile).resize((150, 150), Image.NEAREST)
        comp(img, tile.convert('RGBA'), (x, ty))
        corners(d, (x - 4, ty - 4, x + 154, ty + 154), MINT, a, 18, 3)
        text(d, (x + 176, ty + 52), 'SELECTED', font('mono', 16), MINT, alpha=a)
        text(d, (x + 176, ty + 76), 'PATTERN', font('mono', 16), MINT, alpha=a)
    text(d, (x, VIDEO_BOX[1] + VIDEO_BOX[3] - 64), f'SOURCE {src_t:05.2f} s', font('mono', 18), MINT, alpha=a)
    text(d, (x, VIDEO_BOX[1] + VIDEO_BOX[3] - 36), 'CPU run · native-size inputs', font('mono', 15), MUTED, alpha=a)


def track_overlay(img, track, idx, src_w, f, box=VIDEO_BOX, a=1.0):
    frames = track['frames']
    rec = frames[min(idx, len(frames) - 1)]
    d = D(img)
    if rec['status'] == 'TRACKING':
        # trail of real accepted centres
        pts = [to_panel(r['center'], src_w, box)[:2] for r in frames[max(0, idx - 25):idx + 1]
               if r['status'] == 'TRACKING']
        for k in range(1, len(pts)):
            d.line([pts[k - 1], pts[k]], fill=rgba(MINT, a * 0.25 + 0.6 * a * k / len(pts)), width=3)
        if rec['search_box']:
            sx, sy, sw, sh = rec['search_box']
            p0 = to_panel((sx, sy), src_w, box)
            p1 = to_panel((sx + sw, sy + sh), src_w, box)
            dashed_rect(d, (p0[0], p0[1], p1[0], p1[1]), MINT, a * 0.55, 10, 8, 2, f * 2)
        cx, cy, s = to_panel(rec['center'], src_w, box)
        hw, hh = rec['size'][0] * s / 2 + 6, rec['size'][1] * s / 2 + 6
        layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
        corners(D(layer), (cx - hw, cy - hh, cx + hw, cy + hh), MINT, a, None, 4)
        glow(img, layer, 6)
        d = D(img)
        d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=rgba(MINT, a))
    elif rec['search_box']:
        sx, sy, sw, sh = rec['search_box']
        p0 = to_panel((sx, sy), src_w, box)
        p1 = to_panel((sx + sw, sy + sh), src_w, box)
        dashed_rect(d, (p0[0], p0[1], p1[0], p1[1]), AMBER, a * 0.8, 14, 10, 2, f * 3)
    return rec


def tag(img, x, y, label, color=MINT, a=1.0, size=15):
    d = D(img)
    f = font('monob', size)
    w = f.getlength(label) + 16
    d.rectangle([x, y, x + w, y + size + 12], fill=(4, 7, 10, int(215 * a)), outline=rgba(color, a * 0.8))
    d.text((x + 8, y + (size + 12) / 2 + 1), label, font=f, fill=rgba(color, a), anchor='lm')


# ======================================================================== shot 1: intro
def shot1(f):
    fr = SRC['intro'][min(int(f / FPS * 25), len(SRC['intro']) - 1)]
    img = Image.fromarray(np.asarray(fr)).resize((W, H), Image.BICUBIC).convert('RGB')
    a = clamp(prog(f, 6, 16)) * clamp(1 - prog(f, 120, 135))
    if a > 0:
        tag(img, 40, H - 76, 'ORIGINAL FOOTAGE', TEXT, a, 20)
    return img


# ======================================================================== shot 2: question
BALL_T = 4.5            # intro time of the frozen frame (ball in flight)
BALL_XY = (800, 446)    # ball position in the 1280x720 intro frame (located by hand)


def shot2(f):
    intro = SRC['intro']
    if f < 16:
        # rewind scrub from 12.0 s back to the ball frame
        t = 12.0 - (12.0 - BALL_T) * ease_in_out(f / 15)
        fr = np.asarray(intro[min(int(t * 25), len(intro) - 1)])
        arr = np.asarray(Image.fromarray(fr).resize((W, H), Image.BILINEAR))
        arr = chroma_split(arr, 10 * math.sin(f / 15 * math.pi))
        arr = arr.copy()
        arr[::4] = (arr[::4] * 0.6).astype(np.uint8)
        img = Image.fromarray(arr).convert('RGB')
        d = D(img)
        text(d, (60, H - 80), '◀◀ REWIND', font('monob', 34), TEXT)
        return img
    fr = np.asarray(intro[int(BALL_T * 25)])
    k = ease_in_out(prog(f, 16, 150))
    s = 1.0 + 0.28 * k
    bx, by = BALL_XY[0] * 1.5, BALL_XY[1] * 1.5
    cx, cy = lerp(W / 2, bx, 0.55 * k), lerp(H / 2, by, 0.55 * k)
    arr = zoom(np.asarray(Image.fromarray(fr).resize((W, H), Image.BICUBIC)), s, cx, cy)
    dark = lerp(1.0, 0.42, ease_out(prog(f, 16, 34)))
    img = Image.fromarray((arr * dark).astype(np.uint8)).convert('RGB')
    # ball position on screen after zoom
    px = (bx - cx) * s + W / 2
    py = (by - cy) * s + H / 2
    d = D(img)
    ca = ease_out(prog(f, 22, 40))
    r = lerp(140, 30, back_out(prog(f, 22, 42)))
    layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
    ld = D(layer)
    corners(ld, (px - r, py - r, px + r, py + r), MINT, ca, r * 0.5, 4)
    pulse = (f - 40) % 30 / 30
    if f > 40:
        ring(ld, px, py, 30 + 60 * pulse, MINT, (1 - pulse) * 0.7, 3)
    glow(img, layer, 8)
    d = D(img)
    # typed question
    q1 = 'Can we track that ball...'
    n1 = int(len(q1) * prog(f, 34, 56))
    text(d, (W / 2, 190), q1[:n1], font('semi', 58), TEXT, anchor='mm')
    q2 = 'without machine learning?'
    t2 = prog(f, 66, 80)
    if t2 > 0:
        sc = back_out(t2)
        fsz = 100 * (0.6 + 0.4 * sc)
        text(d, (W / 2, 300), q2, font('black', fsz), MINT, anchor='mm', alpha=clamp(t2 * 2))
    return img


# ======================================================================== shot 3: wordmark
def wordmark(img, f0, cx=W / 2, cy=H / 2, scale=1.0, a=1.0, sub=True):
    d = D(img)
    t = ease_out(prog(f0, 0, 26))
    sp = lerp(60, 14, t) * scale
    f_ = font('black', 170 * scale)
    layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
    ld = D(layer)
    text(ld, (cx, cy), 'FALCON', f_, TEXT, anchor='mm', alpha=a * t, spacing=sp)
    w = text_width('FALCON', f_, sp)
    r = w / 2 + 50 * scale
    corners(ld, (cx - r, cy - 125 * scale, cx + r, cy + 110 * scale), MINT, a * t, 40 * scale, max(2, int(5 * scale)))
    glow(img, layer, 18, 0.6)
    d = D(img)
    if sub:
        ts = ease_out(prog(f0, 18, 36))
        text(d, (cx, cy + 160 * scale), 'SIGNAL 13', font('monob', 34 * scale), MINT, anchor='mm', alpha=a * ts,
             spacing=10 * scale)


def signal_wave(img, f, y=H / 2, a=0.35, amp=60):
    d = D(img)
    pts = []
    for x in range(0, W + 1, 8):
        u = x / W
        v = math.sin(u * 18 + f * 0.25) * 0.6 + math.sin(u * 41 - f * 0.4) * 0.3 + math.cos(u * 7 + f * 0.1) * 0.4
        env = math.exp(-((u - 0.5) / 0.32) ** 2)
        pts.append((x, y + v * amp * env))
    d.line(pts, fill=rgba(MINT, a), width=3)


def shot3(f):
    img = canvas()
    if f < 56:
        signal_wave(img, f, H / 2 + 120, 0.25 * prog(f, 0, 10))
        d = D(img)
        t = back_out(prog(f, 0, 14))
        text(d, (W / 2, H / 2 - 40), 'Yes.', font('black', 150 * (0.7 + 0.3 * t)), TEXT, anchor='mm',
             alpha=clamp(prog(f, 0, 6)) * clamp(1 - prog(f, 48, 56)))
        t2 = ease_out(prog(f, 14, 28))
        text(d, (W / 2, H / 2 + 90 + 30 * (1 - t2)), 'Only signals.', font('bold', 64), MINT, anchor='mm',
             alpha=t2 * clamp(1 - prog(f, 48, 56)))
        # selection corners from shot 2 resolve into the wordmark frame
        return img
    f0 = f - 52
    signal_wave(img, f, H / 2 + 230, 0.3)
    wordmark(img, f0, W / 2, H / 2 - 30)
    if f > 104:
        # zoom-through out of the wordmark
        k = prog(f, 104, 120)
        arr = zoom(np.asarray(img.convert('RGB')), 1 + 1.5 * k ** 2)
        arr = (arr.astype(np.float32) * (1 - 0.85 * k) + np.array(BG) * 0.85 * k).astype(np.uint8)
        img = Image.fromarray(arr).convert('RGB')
    return img


# ======================================================================== shot 4: Messi
def messi_src_idx(f):
    return min(int((f - 75) / FPS * 25), len(SRC['messi']) - 1)


def shot4(f):
    img = canvas()
    header(img, f, '01 / ACTUAL RUN', 'You pick the ball.' if f < 75 else 'The real tracking attempt.')
    trk = TRACK['messi']
    x, y, w, h = trk['roi']
    if f < 75:
        # --- reference image + ROI selection; crop lifts into the side panel
        media_frame(img, REF_PNG)
        d = D(img)
        cx, cy, s = to_panel((x + w / 2, y + h / 2), 1280)
        t = ease_out(prog(f, 8, 30))
        r = lerp(220, w * s / 2 + 8, t)
        layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
        corners(D(layer), (cx - r, cy - r, cx + r, cy + r), MINT, clamp(prog(f, 6, 12)), 22, 4)
        glow(img, layer, 6)
        d = D(img)
        tag(img, 56, 172, 'REFERENCE IMAGE', TEXT, 1.0)
        d = side_panel(img)
        sx0 = SIDE_X + 26
        chip(d, sx0, VIDEO_BOX[1] + 26, 'CHOOSE THE TARGET', MINT)
        text(d, (sx0, VIDEO_BOX[1] + 92), 'A visual pattern.', font('bold', 32), TEXT)
        text(d, (sx0, VIDEO_BOX[1] + 136), 'Not a football detector.', font('reg', 21), MUTED)
        text(d, (sx0, VIDEO_BOX[1] + 164), 'No pretrained model.', font('reg', 21), MUTED)
        lt = ease_out(prog(f, 30, 52))
        tx, ty = lerp(cx - 20, sx0 + 60, lt), lerp(cy - 20, VIDEO_BOX[1] + 250, lt)
        sz = lerp(w * s + 10, 230, lt)
        if f >= 30:
            tile = Image.fromarray(CROP).resize((int(sz), int(sz)), Image.NEAREST).convert('RGB')
            comp(img, tile, (int(tx), int(ty)))
            corners(D(img), (tx - 5, ty - 5, tx + sz + 5, ty + sz + 5), MINT, 1, 26, 3)
        d = D(img)
        text(d, (sx0, VIDEO_BOX[1] + 520), '32 × 32 native-pixel crop', font('mono', 17), MUTED, alpha=prog(f, 50, 60))
        text(d, (sx0, VIDEO_BOX[1] + 548), 'Reference-image selection', font('mono', 17), MINT, alpha=prog(f, 50, 60))
        return img
    if f < 225:
        # --- genuine run at native speed
        idx = messi_src_idx(f)
        media_frame(img, np.asarray(SRC['messi'][idx]))
        rec = track_overlay(img, trk, idx, 1280, f)
        tag(img, 56, 172, 'ACTUAL TRACKER OUTPUT', MINT)
        if rec['status'] == 'SEARCHING':
            tag(img, 56, VIDEO_BOX[1] + VIDEO_BOX[3] - 44, 'GLOBAL SEARCH · 4 TILES / FRAME', AMBER)
        status_block(img, rec, idx / 25, 'messi', live_tile=CROP, lost=idx > 90)
        return img
    if f < 297:
        # --- labelled replay of the accepted interval, 2x zoom, native speed
        idx = min(50 + (f - 225) * 25 // FPS, 124)
        rec = trk['frames'][idx]
        tracked = [r for r in trk['frames'] if r['status'] == 'TRACKING']
        near = min(tracked, key=lambda r: abs(r['i'] - idx))
        cx, cy = near['center']
        z = 2.0
        cw, ch = 1280 / z, 720 / z
        x0 = clamp(cx - cw / 2, 0, 1280 - cw)
        y0 = clamp(cy - ch / 2, 0, 720 - ch)
        fr = Image.fromarray(np.asarray(SRC['messi'][idx])).resize((1400, 788), Image.BICUBIC,
                                                                    box=(x0, y0, x0 + cw, y0 + ch))
        media_frame(img, fr)
        if rec['status'] == 'TRACKING':
            px = VIDEO_BOX[0] + (rec['center'][0] - x0) * 1400 / cw
            py = VIDEO_BOX[1] + (rec['center'][1] - y0) * 788 / ch
            hw = rec['size'][0] * 1400 / cw / 2 + 8
            layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
            corners(D(layer), (px - hw, py - hw, px + hw, py + hw), MINT, 1, None, 4)
            glow(img, layer, 6)
        tag(img, 56, 172, 'REPLAY · 2× EDITORIAL ZOOM', TEXT)
        status_block(img, rec, idx / 25, 'messi', live_tile=CROP, lost=idx > 90)
        return img
    # --- paused on an accepted frame
    idx = 78
    rec = trk['frames'][idx]
    cx, cy = rec['center']
    z = 2.0
    cw, ch = 1280 / z, 720 / z
    x0, y0 = clamp(cx - cw / 2, 0, 1280 - cw), clamp(cy - ch / 2, 0, 720 - ch)
    fr = Image.fromarray(np.asarray(SRC['messi'][idx])).resize((1400, 788), Image.BICUBIC,
                                                                box=(x0, y0, x0 + cw, y0 + ch))
    fr = Image.fromarray((np.asarray(fr) * 0.8).astype(np.uint8))
    media_frame(img, fr)
    px = VIDEO_BOX[0] + (cx - x0) * 1400 / cw
    py = VIDEO_BOX[1] + (cy - y0) * 788 / ch
    hw = rec['size'][0] * 1400 / cw / 2 + 8
    layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
    ld = D(layer)
    corners(ld, (px - hw, py - hw, px + hw, py + hw), MINT, 1, None, 4)
    pulse = (f - 297) % 24 / 24
    ring(ld, px, py, hw + 40 * pulse, MINT, 1 - pulse, 2)
    glow(img, layer, 6)
    tag(img, 56, 172, 'PAUSED FOR INSPECTION · 2× ZOOM', TEXT)
    d = side_panel(img)
    sx0 = SIDE_X + 26
    chip(d, sx0, VIDEO_BOX[1] + 26, 'MEASURED RESULT', MINT)
    text(d, (sx0, VIDEO_BOX[1] + 92), 'Brief, real lock.', font('bold', 34), TEXT)
    text(d, (sx0, VIDEO_BOX[1] + 140), 'Tracked about 1.2 s', font('reg', 23), MUTED)
    text(d, (sx0, VIDEO_BOX[1] + 170), '(source 2.48 – 3.60 s).', font('reg', 23), MUTED)
    text(d, (sx0, VIDEO_BOX[1] + 200), 'Then the match fails.', font('reg', 23), AMBER)
    text(d, (sx0, VIDEO_BOX[1] + 260), 'THIS FRAME', font('mono', 16), MUTED)
    text(d, (sx0, VIDEO_BOX[1] + 286), f"PSR {rec['psr']:.1f}   NCC {rec['ncc']:.2f}", font('monob', 26), TEXT)
    full = Image.fromarray(np.asarray(SRC['messi'][idx])).resize((360, 202), Image.BILINEAR)
    comp(img, full.convert('RGBA'), (sx0, VIDEO_BOX[1] + 350))
    d = D(img)
    rx, ry = sx0 + x0 * 360 / 1280, VIDEO_BOX[1] + 350 + y0 * 202 / 720
    d.rectangle([rx, ry, rx + cw * 360 / 1280, ry + ch * 202 / 720], outline=rgba(MINT, 0.9), width=2)
    text(d, (sx0, VIDEO_BOX[1] + 566), 'Full-frame context', font('mono', 15), MUTED)
    text(d, (sx0, VIDEO_BOX[1] + VIDEO_BOX[3] - 64), f'SOURCE {idx / 25:05.2f} s', font('mono', 18), MINT)
    return img


# ======================================================================== shot 5: waves
def custom_fft_reconstruct(gray, keep):
    """Rebuild a picture from its `keep` strongest waves using the repo's custom FFT."""
    from signal13.core.fft_engine import fft_2d, ifft_2d
    spec = fft_2d(gray.astype(np.float32))
    mag = np.abs(spec).ravel()
    order = np.argsort(-mag, kind='stable')
    mask = np.zeros(mag.size, bool)
    mask[order[:keep]] = True
    rec = np.real(ifft_2d(spec * mask.reshape(spec.shape)))
    return rec


_RECON = {}


def recon_tile(keep):
    if keep not in _RECON:
        cr = CROP.astype(np.float32)
        out = np.stack([custom_fft_reconstruct(cr[..., c], keep) for c in range(3)], -1)
        _RECON[keep] = np.clip(out, 0, 255).astype(np.uint8)
    return _RECON[keep]


WAVES = [  # (kind, cycles across strip, amplitude, phase)  fixed, seed-free
    ('sin', 1.0, 1.0, 0.0), ('cos', 2.0, 0.7, 0.4), ('sin', 3.0, 0.5, 1.1),
    ('cos', 5.0, 0.35, 0.2), ('sin', 8.0, 0.22, 2.0),
]


def wave_y(kind, cyc, amp, ph, u, f):
    a = 2 * math.pi * cyc * u + ph + f * 0.06 * cyc
    return amp * (math.sin(a) if kind == 'sin' else math.cos(a))


def heat(surface):
    s = (surface - surface.min()) / (np.ptp(surface) + 1e-9)
    s = s ** 3.0
    c0, c1, c2 = np.array([8, 14, 20.]), np.array(MINT, float), np.array([255, 255, 255.])
    lo = np.clip(s * 2, 0, 1)[..., None]
    hi = np.clip(s * 2 - 1, 0, 1)[..., None]
    rgb = c0 * (1 - lo) + c1 * lo
    rgb = rgb * (1 - hi) + c2 * hi
    return rgb.astype(np.uint8)


def shot5(f):
    img = canvas()
    header(img, f, '02 / HOW IT SEES', 'A picture is just waves.')
    d = D(img)
    chip(d, 1880, 82, 'CONCEPTUAL', AMBER, 1, 17, anchor='r')
    if f < 200:
        # left: selected crop, below it the rebuild from waves
        ta = ease_out(prog(f, 0, 14))
        tx, ty, ts = 120, 200, 300
        tile = Image.fromarray(CROP).resize((ts, ts), Image.NEAREST).convert('RGB')
        comp(img, tile, (tx, int(ty + 30 * (1 - ta))))
        d = D(img)
        corners(d, (tx - 6, ty - 6, tx + ts + 6, ty + ts + 6), MINT, ta, 30, 3)
        text(d, (tx, ty + ts + 22), 'Selected crop', font('mono', 18), MUTED, alpha=ta)
        # wave strips
        x0, x1 = 560, 1820
        for k, (kind, cyc, amp, ph) in enumerate(WAVES):
            t = ease_out(prog(f, 18 + k * 9, 44 + k * 9))
            if t <= 0:
                continue
            yy = 220 + k * 92
            xe = lerp(x0, x1, t)
            pts = [(x, yy + 30 * wave_y(kind, cyc, amp, ph, (x - x0) / (x1 - x0), f))
                   for x in np.arange(x0, xe + 1, 6)]
            if len(pts) > 1:
                d.line(pts, fill=rgba(mix(MINT, (120, 170, 255), k / 5), 0.9), width=3)
            text(d, (x0 - 20, yy), kind, font('mono', 18), MUTED, anchor='rm', alpha=t)
        # their sum
        ts_ = ease_out(prog(f, 80, 110))
        if ts_ > 0:
            yy = 720
            xe = lerp(x0, x1, ts_)
            pts = [(x, yy + 36 * sum(wave_y(*w_, (x - x0) / (x1 - x0), f) for w_ in WAVES))
                   for x in np.arange(x0, xe + 1, 5)]
            layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
            D(layer).line(pts, fill=rgba(TEXT, 1), width=4)
            glow(img, layer, 8, 0.8)
            d = D(img)
            text(d, (x0 - 20, yy), 'sum', font('monob', 18), TEXT, anchor='rm', alpha=ts_)
            chip(d, x0, 800, 'SINE + COSINE WAVES', MINT, ts_)
        # rebuild tile
        tr = prog(f, 70, 190)
        if tr > 0:
            keep = int(round(1 + (1024 - 1) * tr ** 2.2))
            rt = Image.fromarray(recon_tile(keep)).resize((ts, ts), Image.NEAREST).convert('RGB')
            ry = 600
            comp(img, rt, (tx, ry))
            d = D(img)
            corners(d, (tx - 6, ry - 6, tx + ts + 6, ry + ts + 6), MINT, 1, 30, 3)
            text(d, (tx, ry + ts + 18), f'Rebuilt from {keep} waves', font('mono', 18), MINT)
            text(d, (tx + ts + 18, ry + 10), 'Custom FFT', font('mono', 15), MUTED)
        if f < 10:
            k = 1 - prog(f, 0, 10)
            arr = shift_x(motion_blur_x(np.asarray(img.convert('RGB')), 70 * k), W * 0.2 * k ** 2)
            img = Image.fromarray(arr).convert('RGB')
        if f > 188:
            k = prog(f, 188, 200)
            arr = shift_x(motion_blur_x(np.asarray(img.convert('RGB')), 60 * k), -W * 0.15 * k ** 2)
            img = Image.fromarray(arr).convert('RGB')
        return img
    # ---- compare: template waves vs search patch, best match = brightest peak
    f2 = f - 200
    d = D(img)
    ta = ease_out(prog(f2, 0, 16))
    s1 = Image.fromarray(CROP).resize((220, 220), Image.NEAREST).convert('RGB')
    comp(img, s1, (int(80 + 40 * (1 - ta)), 260))
    win = DIAG['Windowed_search']
    wn = ((win - win.min()) / (np.ptp(win) + 1e-9) * 255).astype(np.uint8)
    s2 = Image.fromarray(wn).resize((330, 330), Image.BILINEAR).convert('RGB')
    comp(img, s2, (int(400 + 40 * (1 - ta)), 205))
    d = D(img)
    text(d, (80, 500), 'Selected pattern', font('mono', 18), MUTED, alpha=ta)
    text(d, (400, 556), 'Nearby patch', font('mono', 18), MUTED, alpha=ta)
    text(d, (340, 370), '×', font('bold', 54), MINT, anchor='mm', alpha=ta)
    # the two wave bundles slide together
    tm = ease_in_out(prog(f2, 14, 60))
    for k, (kind, cyc, amp, ph) in enumerate(WAVES[:4]):
        yy = 680 + k * 46
        xa = 80
        xb = 760
        off = (1 - tm) * 1.4
        pa = [(x, yy + 16 * wave_y(kind, cyc, amp, ph, (x - xa) / (xb - xa), 0)) for x in range(xa, xb + 1, 6)]
        pb = [(x, yy + 16 * wave_y(kind, cyc, amp, ph + off * (k + 1), (x - xa) / (xb - xa), 0))
              for x in range(xa, xb + 1, 6)]
        d.line(pa, fill=rgba(MINT, 0.85 * ta), width=3)
        d.line(pb, fill=rgba((120, 170, 255), 0.85 * ta), width=3)
    text(d, (80, 892), 'Line the waves up…', font('reg', 22), MUTED, alpha=ta)
    # correlation surface (actual diagnostic array) revealed by a sweep
    surf = np.roll(DIAG['Correlation_surface'], (64, 64), axis=(0, 1))  # display: zero shift at centre
    hm = Image.fromarray(heat(surf)).resize((520, 520), Image.BICUBIC).convert('RGB')
    hx, hy = 1000, 200
    sw = ease_in_out(prog(f2, 50, 95))
    if sw > 0:
        cropw = max(1, int(520 * sw))
        comp(img, hm.crop((0, 0, cropw, 520)), (hx, hy))
        d = D(img)
        d.line([(hx + cropw, hy), (hx + cropw, hy + 520)], fill=rgba(TEXT, 0.8 * (1 - prog(f2, 92, 96))), width=3)
        d.rectangle([hx - 2, hy - 2, hx + 522, hy + 522], outline=rgba(BORDER, 1), width=2)
        tag(img, hx + 10, hy + 10, 'ACTUAL DIAGNOSTIC · MESSI FRAME 78', MINT, 1, 13)
    pk = np.unravel_index(np.argmax(surf), surf.shape)
    px, py = hx + (pk[1] + 0.5) * 520 / surf.shape[1], hy + (pk[0] + 0.5) * 520 / surf.shape[0]
    tp = prog(f2, 96, 112)
    if tp > 0:
        layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
        ld = D(layer)
        r = lerp(90, 22, back_out(tp))
        ring(ld, px, py, r, TEXT, tp, 4)
        pulse = (f2 - 112) % 26 / 26 if f2 > 112 else 0
        if f2 > 112:
            ring(ld, px, py, 22 + 50 * pulse, MINT, 1 - pulse, 3)
        glow(img, layer, 10)
        d = D(img)
        text(d, (hx + 260, hy + 560), 'Best match = brightest peak', font('bold', 34), TEXT, anchor='mm', alpha=tp)
        text(d, (hx + 260, hy + 604), f"PSR {float(DIAG['psr']):.1f} · peak is sharp and distinctive",
             font('mono', 17), MUTED, anchor='mm', alpha=tp)
        # cross-section through the peak: a wave with a spike
        row = surf[pk[0]]
        row = (row - row.min()) / (np.ptp(row) + 1e-9)
        pts = [(hx + 560 + i * 300 / len(row), hy + 470 - 220 * v) for i, v in enumerate(row)]
        d.line(pts, fill=rgba(MINT, tp), width=3)
        text(d, (hx + 560, hy + 500), 'slice through peak', font('mono', 14), MUTED, alpha=tp)
    if f > 348:
        k = prog(f, 348, 360)
        arr = shift_x(motion_blur_x(np.asarray(img.convert('RGB')), 70 * k), -W * 0.2 * k ** 2)
        img = Image.fromarray(arr).convert('RGB')
    if f2 < 10:
        k = 1 - prog(f2, 0, 10)
        arr = shift_x(motion_blur_x(np.asarray(img.convert('RGB')), 60 * k), W * 0.15 * k ** 2)
        img = Image.fromarray(arr).convert('RGB')
    return img


# ======================================================================== shot 6: predict / verify
PAST = [(420, 700), (560, 610), (700, 535), (840, 478), (980, 440)]
PRED = (1130, 418)


def shot6(f):
    img = canvas()
    header(img, f, '03 / PREDICT', 'Predict, then verify.')
    d = D(img)
    chip(d, 1880, 82, 'CONCEPTUAL', AMBER, 1, 17, anchor='r')
    for k, p in enumerate(PAST):
        t = ease_out(prog(f, 4 + k * 6, 14 + k * 6))
        if t > 0:
            r = 9 * t
            d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=rgba(MINT, 0.35 + 0.13 * k))
            if k:
                d.line([PAST[k - 1], p], fill=rgba(MINT, 0.5 * t), width=3)
    text(d, (420, 750), 'Past positions', font('mono', 18), MUTED, alpha=prog(f, 10, 20))
    tp = ease_out(prog(f, 36, 60))
    if tp > 0:
        e = (lerp(PAST[-1][0], PRED[0], tp), lerp(PAST[-1][1], PRED[1], tp))
        dashed_line(d, PAST[-1], e, rgba(TEXT, 0.9), 12, 9, 3)
        chip(d, 1020, 320, 'KALMAN PREDICTION', MINT, tp, 16)
    tb = back_out(prog(f, 58, 78))
    if prog(f, 58, 78) > 0:
        r = 70 * tb
        col = MINT if f >= 118 else TEXT
        if f < 118:
            dashed_rect(d, (PRED[0] - r, PRED[1] - r, PRED[0] + r, PRED[1] + r), col, 1, 10, 7, 3, f * 2)
        else:
            layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
            corners(D(layer), (PRED[0] - r, PRED[1] - r, PRED[0] + r, PRED[1] + r), MINT, 1, 26, 5)
            glow(img, layer, 10)
            d = D(img)
        text(d, (PRED[0], PRED[1] + 96), 'Small search region', font('mono', 17), MUTED, anchor='mm')
    ti = ease_out(prog(f, 78, 98))
    if ti > 0:
        sx, sy = lerp(1500, PRED[0] - 32, ti), lerp(250, PRED[1] - 32, ti)
        tile = Image.fromarray(CROP).resize((64, 64), Image.NEAREST).convert('RGB')
        comp(img, tile, (int(sx), int(sy)))
        d = D(img)
    for k, lab in enumerate(('PSR ✓  sharp peak', 'NCC ✓  looks the same')):
        t = back_out(prog(f, 100 + k * 10, 114 + k * 10))
        if prog(f, 100 + k * 10, 114 + k * 10) > 0:
            chip(d, 1300, 520 + k * 60, lab, MINT, clamp(t), 20)
    tv = ease_out(prog(f, 118, 132))
    text(d, (1300, 680), 'Verified.', font('black', 64), TEXT, alpha=tv)
    text(d, (1300, 772), 'PSR + NCC checks', font('mono', 17), MUTED, alpha=tv)
    if f < 10:
        k = 1 - prog(f, 0, 10)
        arr = shift_x(motion_blur_x(np.asarray(img.convert('RGB')), 70 * k), W * 0.2 * k ** 2)
        img = Image.fromarray(arr).convert('RGB')
    if f > 168:
        k = prog(f, 168, 180)
        arr = shift_x(motion_blur_x(np.asarray(img.convert('RGB')), 70 * k), -W * 0.2 * k ** 2)
        img = Image.fromarray(arr).convert('RGB')
    return img


# ======================================================================== shot 7: search again
GRID_C, GRID_R = 10, 5
SCAN_START, SCAN_EVERY, SCAN_BATCH = 56, 10, 4   # 4 tiles per step, checked in order (like the tracker)
TARGET_TILE = 37                                  # where the illustrated target sits
FOUND_AT = SCAN_START + SCAN_EVERY * (TARGET_TILE // SCAN_BATCH)


def shot7(f):
    img = canvas()
    header(img, f, '04 / SEARCH', 'Sliding-window search.')
    d = D(img)
    chip(d, 1880, 82, 'CONCEPTUAL', AMBER, 1, 17, anchor='r')
    gx0, gy0, gx1, gy1 = 120, 200, 1360, 860
    cx, cy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
    if f < 34:
        r = 70
        a = 1 - prog(f, 22, 34)
        dashed_rect(d, (cx - r, cy - r, cx + r, cy + r), AMBER, a * (0.5 + 0.5 * math.cos(f * 0.6)), 10, 7, 3)
        text(d, (cx, cy + 110), 'Confidence drops  →  widen the search', font('monob', 26), AMBER, anchor='mm', alpha=a)
    tg = ease_out(prog(f, 26, 52))
    found = f >= FOUND_AT
    if tg > 0:
        cw, ch = (gx1 - gx0) / GRID_C, (gy1 - gy0) / GRID_R
        step = (f - SCAN_START) // SCAN_EVERY if f >= SCAN_START else -1
        if found:
            step = TARGET_TILE // SCAN_BATCH
        window = set(range(step * SCAN_BATCH, step * SCAN_BATCH + SCAN_BATCH)) if step >= 0 else set()
        for i in range(GRID_R):
            for j in range(GRID_C):
                k = i * GRID_C + j
                x0 = lerp(cx, gx0 + j * cw, tg) + 4
                y0 = lerp(cy, gy0 + i * ch, tg) + 4
                x1, y1 = x0 + (cw - 8) * tg, y0 + (ch - 8) * tg
                checked = step >= 0 and k < step * SCAN_BATCH
                if k == TARGET_TILE and found:
                    continue
                if k in window:
                    fill, out, wd = rgba(MINT, 0.22), rgba(MINT, 0.95), 3
                elif checked:
                    fill, out, wd = rgba(MINT, 0.06), rgba(MINT_DIM, 0.6), 2
                else:
                    fill, out, wd = rgba(PANEL, 0.95), rgba(BORDER, 1), 2
                d.rounded_rectangle([x0, y0, x1, y1], radius=6, fill=fill, outline=out, width=wd)
        # the sliding window frame itself, gliding between steps
        if step >= 0 and not found:
            t = ease_in_out(prog((f - SCAN_START) % SCAN_EVERY, 0, 5))
            k0 = step * SCAN_BATCH
            prev = max(k0 - SCAN_BATCH, 0)
            def box(k):
                i, j = divmod(k, GRID_C)
                n = min(SCAN_BATCH, GRID_C - j)
                return (gx0 + j * cw, gy0 + i * ch, gx0 + (j + n) * cw, gy0 + (i + 1) * ch)
            b0, b1 = box(prev), box(k0)
            if b0[1] != b1[1]:
                b0 = b1
            bb = tuple(lerp(p, q, t) for p, q in zip(b0, b1))
            layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
            corners(D(layer), (bb[0] - 4, bb[1] - 4, bb[2] + 4, bb[3] + 4), MINT, 1, 24, 4)
            glow(img, layer, 8)
            d = D(img)
        # the target tile
        i, j = divmod(TARGET_TILE, GRID_C)
        tx0, ty0 = gx0 + j * cw + 4, gy0 + i * ch + 4
        if found:
            tf = back_out(prog(f, FOUND_AT, FOUND_AT + 14))
            layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
            ld = D(layer)
            ld.rounded_rectangle([tx0, ty0, tx0 + cw - 8, ty0 + ch - 8], radius=6, fill=rgba(MINT, 0.3),
                                 outline=rgba(MINT, 1), width=4)
            pulse = (f - FOUND_AT) % 24 / 24
            ring(ld, tx0 + (cw - 8) / 2, ty0 + (ch - 8) / 2, 50 + 70 * pulse, MINT, 1 - pulse, 3)
            glow(img, layer, 12)
            sz = max(2, int(84 * clamp(tf, 0.01, 1.2)))
            tile = Image.fromarray(CROP).resize((sz, sz), Image.NEAREST)
            comp(img, tile, (tx0 + (cw - 8 - sz) / 2, ty0 + (ch - 8 - sz) / 2))
            d = D(img)
        text(d, (gx0, gy1 + 22), 'Illustrated · tiles checked in order, a few per frame · fixed reference template',
             font('mono', 16), MUTED, alpha=tg)
    sx = 1430
    t1 = ease_out(prog(f, 40, 56))
    if not found and t1 > 0.01:
        chip(d, sx, 220, 'SEARCHING', AMBER, t1, 20)
    else:
        chip(d, sx, 220, 'TRACKING', MINT, 1, 20)
    text(d, (sx, 290), 'Slide and scan.', font('black', 54), TEXT, alpha=t1)
    text(d, (sx, 360), 'Tile by tile,', font('reg', 26), MUTED, alpha=t1)
    text(d, (sx, 394), 'across the whole frame.', font('reg', 26), MUTED, alpha=t1)
    text(d, (sx, 428), 'It keeps going.', font('reg', 26), MUTED, alpha=t1)
    t2 = ease_out(prog(f, FOUND_AT, FOUND_AT + 14))
    text(d, (sx, 510), 'Target found.', font('semi', 34), MINT, alpha=t2)
    text(d, (sx, 552), 'Back to tracking.', font('semi', 28), TEXT, alpha=t2)
    if f < 10:
        k = 1 - prog(f, 0, 10)
        arr = shift_x(motion_blur_x(np.asarray(img.convert('RGB')), 70 * k), W * 0.2 * k ** 2)
        img = Image.fromarray(arr).convert('RGB')
    return img


# ======================================================================== shot 8: frisbee
def shot8(f):
    img = canvas()
    trk = TRACK['frisbee']
    header(img, f, '05 / NEW TARGET', 'New target: frisbee.')
    src = SRC['frisbee']
    n = len(src)
    if f < 36:
        media_frame(img, np.asarray(src[0]))
        x, y, w, h = trk['roi']
        cx, cy, s = to_panel((x + w / 2, y + h / 2), 640)
        t = ease_out(prog(f, 4, 26))
        r = lerp(200, w * s / 2 + 8, t)
        layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
        corners(D(layer), (cx - r, cy - r, cx + r, cy + r), MINT, clamp(prog(f, 2, 8)), 22, 4)
        glow(img, layer, 6)
        tag(img, 56, 172, 'SELECTED DISC · DEFAULT CONFIGURATION', TEXT)
        d = side_panel(img)
        chip(d, SIDE_X + 26, VIDEO_BOX[1] + 26, 'CHOOSE THE TARGET', MINT)
        text(d, (SIDE_X + 26, VIDEO_BOX[1] + 92), 'A new pattern.', font('bold', 32), TEXT)
        text(d, (SIDE_X + 26, VIDEO_BOX[1] + 136), 'Small. Spinning.', font('reg', 22), MUTED)
        text(d, (SIDE_X + 26, VIDEO_BOX[1] + 164), 'Changing size.', font('reg', 22), MUTED)
        return img
    paused = f >= 36 + n
    idx = min(int((f - 36) / FPS * trk['fps']), n - 1)
    media_frame(img, np.asarray(src[idx]))
    rec = track_overlay(img, trk, idx, 640, f)
    lost_at = next(r['i'] for r in trk['frames'] if r['i'] > 0 and r['status'] == 'SEARCHING')
    lost = idx >= lost_at
    tag(img, 56, 172, 'ACTUAL TRACKER OUTPUT' + (' · PAUSED ON FINAL FRAME' if paused else ''), TEXT if paused else MINT)
    if lost:
        tl = prog(idx, lost_at, lost_at + 8)
        tag(img, 56, VIDEO_BOX[1] + VIDEO_BOX[3] - 48, 'MATCH LOST · APPEARANCE CHANGED AT THE CATCH', RED, tl, 17)
    tile = np.asarray(src[0])[trk['roi'][1]:trk['roi'][1] + trk['roi'][3], trk['roi'][0]:trk['roi'][0] + trk['roi'][2]]
    status_block(img, rec, idx / trk['fps'], 'frisbee', live_tile=tile, lost=lost)
    if f > 290:
        k = prog(f, 290, 300)
        arr = (np.asarray(img.convert('RGB')) * (1 - k)).astype(np.uint8)
        img = Image.fromarray(arr).convert('RGB')
    return img


# ======================================================================== shot 9: close + authors
def shot9(f):
    img = canvas()
    d = D(img)
    a_in = prog(f, 0, 12)
    text(d, (40, 34), 'SIGNAL 13', font('mono', 15), MINT, alpha=a_in)
    text(d, (1880, 34), 'CSE 220 · BUET', font('mono', 15), MUTED, anchor='ra', alpha=a_in)
    if f < 176:
        signal_wave(img, f, 860, 0.22 * a_in, 40)
        wordmark(img, f, W / 2, 420, 1.0, a_in, sub=False)
        d = D(img)
        tl = 'Select. Predict. Verify. Search again.'
        n = int(len(tl) * prog(f, 26, 70))
        text(d, (W / 2, 640), tl[:n], font('semi', 44), MINT, anchor='mm')
        t3 = ease_out(prog(f, 72, 90))
        text(d, (W / 2, 712), 'No machine learning  ·  Custom FFT  +  Kalman filter  ·  CPU-based',
             font('mono', 22), MUTED, anchor='mm', alpha=t3)
        if f > 160:
            k = prog(f, 160, 176)
            arr = np.asarray(img.convert('RGB')).astype(np.float32)
            arr = zoom(arr.astype(np.uint8), 1 + 0.08 * k).astype(np.float32)
            arr = arr * (1 - k) + np.array(BG) * k
            img = Image.fromarray(arr.astype(np.uint8)).convert('RGB')
        return img
    # ---- authors (settles by local frame 330; last 60 frames are still)
    f3 = f - 176
    signal_wave(img, 150, 980, 0.14, 26)
    wordmark(img, 60, W / 2, 170, 0.42, 1.0, sub=False)
    d = D(img)
    text(d, (W / 2, 278), 'FALCON  ·  Signal 13', font('mono', 22), MINT, anchor='mm', alpha=ease_out(prog(f3, 0, 14)))
    text(d, (W / 2, 328), 'Built by', font('reg', 28), MUTED, anchor='mm', alpha=ease_out(prog(f3, 6, 20)))
    for k, (name, sid, pic) in enumerate(AUTHORS):
        t = ease_out(prog(f3, 14 + k * 12, 44 + k * 12))
        if t <= 0:
            continue
        cx = W / 2 + (-330 if k == 0 else 330)
        cy = 420 + 60 * (1 - t)
        size = 360
        x0, y0 = int(cx - size / 2), int(cy)
        card = pic.convert('RGB')
        mask = rounded_mask(size, size, 28)
        if t < 1:
            mask = Image.fromarray((np.asarray(mask) * t).astype(np.uint8))
        layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
        ld = D(layer)
        ld.rounded_rectangle([x0 - 8, y0 - 8, x0 + size + 8, y0 + size + 8], radius=34, outline=rgba(MINT, t), width=4)
        glow(img, layer, 16, 0.9)
        img.paste(card, (x0, y0), mask)
        d = D(img)
        corners(d, (x0 - 22, y0 - 22, x0 + size + 22, y0 + size + 22), MINT, t, 34, 3)
        text(d, (cx, y0 + size + 52), name, font('bold', 40), TEXT, anchor='mm', alpha=t)
        text(d, (cx, y0 + size + 98), sid, font('mono', 24), MINT, anchor='mm', alpha=t, spacing=4)
    return img


SHOT_FN = {'s1': shot1, 's2': shot2, 's3': shot3, 's4': shot4, 's5': shot5, 's6': shot6, 's7': shot7,
           's8': shot8, 's9': shot9}


def render(f):
    for s in SB['shots']:
        if s['start'] <= f < s['end']:
            img = SHOT_FN[s['id']](f - s['start'])
            break
    if f >= 360:
        progress_bar(img, f, TOTAL)
        draw_captions(img, f)
    return np.asarray(img.convert('RGB'))


def _render_bytes(f):
    return render(f).tobytes()


def srt_time(t):
    ms = int(round(t * 1000))
    return f'{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}'


def write_captions(work):
    (work / 'captions.json').write_text(json.dumps(CAPS, indent=1))
    lines = []
    for k, c in enumerate(CAPS):
        nxt = CAPS[k + 1]['start'] if k + 1 < len(CAPS) else TOTAL / FPS
        end = min(c['end'] + 0.45, nxt - 0.02)
        lines += [str(k + 1), f"{srt_time(c['start'])} --> {srt_time(end)}", c['text'], '']
    (HERE / 'captions.srt').write_text('\n'.join(lines))


def main():
    work = Path(sys.argv[1])
    init(work)
    write_captions(work)
    args = sys.argv[2:]
    if args and args[0] == '--stills':
        for f in map(int, args[1].split(',')):
            Image.fromarray(render(f)).save(work / f'still_{f:04d}.png')
        return
    a, b = 0, TOTAL
    if args and args[0] == '--frames':
        a, b = map(int, args[1].split(':'))
    out = work / ('picture.mp4' if (a, b) == (0, TOTAL) else f'picture_{a}_{b}.mp4')
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                            '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium',
                            '-crf', '17', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)],
                           stdin=subprocess.PIPE)
    with Pool(4, initializer=init, initargs=(str(work),)) as pool:
        for k, buf in enumerate(pool.imap(_render_bytes, range(a, b), chunksize=4)):
            enc.stdin.write(buf)
            if k % 100 == 0:
                print('frame', a + k, flush=True)
    enc.stdin.close()
    enc.wait()
    print('wrote', out)


if __name__ == '__main__':
    main()
