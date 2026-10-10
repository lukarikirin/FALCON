"""Build the 80.000 s stereo soundtrack (48 kHz).

0-12 s   : intro.mp4 native commentary, loudness-matched to -16 LUFS, 8-frame fade.
12-80 s  : narration (Kokoro takes placed by the caption schedule) at -16 LUFS,
           a quiet synthesized pad that ducks under the voice, and transition SFX.
Usage: python video_production/audio_mix.py <work_dir>
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SR = 48000
DUR = 80.0
N = int(SR * DUR)
rng = np.random.default_rng(13)


def load(path, start=None, dur=None, mono=False):
    cmd = ['ffmpeg', '-v', 'error']
    if start is not None:
        cmd += ['-ss', str(start)]
    if dur is not None:
        cmd += ['-t', str(dur)]
    cmd += ['-i', str(path), '-f', 'f32le', '-ac', '1' if mono else '2', '-ar', str(SR), '-']
    a = np.frombuffer(subprocess.check_output(cmd), np.float32)
    return a if mono else a.reshape(-1, 2)


def lufs(x, work):
    p = work / '_meas.f32'
    np.ascontiguousarray(x, np.float32).tofile(p)
    out = subprocess.run(['ffmpeg', '-hide_banner', '-f', 'f32le', '-ar', str(SR), '-ac', str(x.shape[1] if x.ndim == 2 else 1),
                          '-i', str(p), '-af', 'loudnorm=print_format=json', '-f', 'null', '-'],
                         capture_output=True, text=True).stderr
    m = json.loads(re.search(r'\{[^{}]*"input_i"[^{}]*\}', out, re.S).group(0))
    return float(m['input_i']), float(m['input_tp'])


def gain_to(x, target, work):
    i, _ = lufs(x, work)
    return x * 10 ** ((target - i) / 20)


def env(n, a, r):
    e = np.ones(n, np.float32)
    a, r = int(a * SR), int(r * SR)
    if a:
        e[:a] = np.linspace(0, 1, a)
    if r:
        e[-r:] *= np.linspace(1, 0, r)
    return e


def whoosh(dur=0.45, up=True, gain=0.25):
    n = int(dur * SR)
    noise = rng.standard_normal(n).astype(np.float32)
    t = np.linspace(0, 1, n)
    # moving band: difference of two low-passes with swept cut-offs (cheap approximation)
    sweep = (t if up else 1 - t)
    hp = noise - np.convolve(noise, np.ones(9) / 9, 'same')
    out = hp * (0.25 + 0.75 * sweep) + np.convolve(noise, np.ones(40) / 40, 'same') * 3 * (1 - sweep)
    shape = np.sin(np.pi * t) ** 2
    return (out * shape * gain).astype(np.float32)


def tone(freq, dur, gain=0.2, decay=8.0, harm=(1, 0.3, 0.1)):
    t = np.arange(int(dur * SR)) / SR
    x = sum(h * np.sin(2 * np.pi * freq * (k + 1) * t) for k, h in enumerate(harm))
    return (x * np.exp(-decay * t) * gain * env(len(t), 0.004, 0.02)).astype(np.float32)


def boom(gain=0.5):
    t = np.arange(int(1.4 * SR)) / SR
    f = 40 + 70 * np.exp(-t * 9)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return (np.sin(ph) * np.exp(-t * 3.2) * gain).astype(np.float32)


def rewind_sfx(dur=0.55, gain=0.3):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 900 * (1 - t / dur) ** 2 + 120
    ph = 2 * np.pi * np.cumsum(f) / SR
    warble = np.sin(ph) * 0.5 + np.sin(ph * 2.01) * 0.25
    noise = np.convolve(rng.standard_normal(n), np.ones(6) / 6, 'same') * 0.6
    return ((warble + noise) * np.sin(np.pi * t / dur) * gain).astype(np.float32)


def pad(n, start_t):
    """Am - F - C - G pad, 4 s per chord, soft saw stack, low-passed."""
    t = np.arange(n) / SR + start_t
    chords = [(57, 60, 64), (53, 57, 60), (48, 55, 64), (55, 59, 62)]
    out = np.zeros(n, np.float32)
    seg = 4.0
    for k in range(int(np.ceil(n / SR / seg)) + 1):
        a = int(k * seg * SR)
        b = min(n, int((k + 1) * seg * SR) + int(0.6 * SR))
        if a >= n:
            break
        tt = t[a:b]
        e = env(b - a, 0.8, 0.9)
        for m in chords[k % 4]:
            fr = 440 * 2 ** ((m - 69) / 12)
            for det in (-0.12, 0.12):
                ph = 2 * np.pi * fr * (1 + det / 100) * tt
                out[a:b] += (np.sin(ph) + 0.33 * np.sin(3 * ph) + 0.2 * np.sin(5 * ph)) * e * 0.08
        root = 440 * 2 ** ((chords[k % 4][0] - 12 - 69) / 12)
        out[a:b] += np.sin(2 * np.pi * root * tt) * e * 0.12
    # slow pulse to give it motion (96 bpm eighths)
    pulse = 0.75 + 0.25 * np.cos(2 * np.pi * t * 96 / 60 * 2) ** 8
    return out * pulse


def place(bus, clip, t, gain=1.0):
    a = int(round(t * SR))
    if clip.ndim == 1:
        clip = np.stack([clip, clip], 1)
    b = min(len(bus), a + len(clip))
    bus[a:b] += clip[:b - a] * gain


def main():
    work = Path(sys.argv[1])
    caps = json.loads((work / 'captions.json').read_text())
    out = np.zeros((N, 2), np.float32)

    # ---- intro: native audio, loudness matched, fade over the last 8 frames
    intro = load(ROOT / 'input_video/intro.mp4', 0, 12.0)
    intro = gain_to(intro, -16.0, work)[:int(12.0 * SR)]
    intro *= env(len(intro), 0, 8 / 30)[:, None]
    out[:len(intro)] += intro

    # ---- narration bus
    vo = np.zeros((N, 2), np.float32)
    for c in caps:
        place(vo, load(work / 'vo' / c['wav'], mono=True), c['start'])
    voiced = np.abs(vo[:, 0]) > 1e-4
    vo = gain_to(vo[int(12 * SR):], -16.0, work)
    out[int(12 * SR):] += vo

    # ---- music bed with ducking
    m0 = int(12.0 * SR)
    music = pad(N - m0, 12.0)
    act = np.convolve(voiced[m0:].astype(np.float32), np.ones(int(0.25 * SR)) / (0.25 * SR), 'same')
    duck = 1 - 0.6 * np.clip(act * 3, 0, 1)
    duck = np.convolve(duck, np.ones(int(0.12 * SR)) / (0.12 * SR), 'same')
    music *= duck * env(len(music), 1.2, 2.5)
    music_st = np.stack([music, np.roll(music, int(0.011 * SR))], 1)
    music_st = gain_to(music_st, -30.0, work)
    out[m0:] += music_st

    # ---- sound design (times in seconds; frames / 30)
    sfx = np.zeros((N, 2), np.float32)
    place(sfx, rewind_sfx(), 12.0)
    for k in range(10):                                   # typing ticks for the question
        place(sfx, tone(2400, 0.03, 0.05, 90), 13.13 + k * 0.075)
    place(sfx, tone(880, 0.6, 0.12, 6), 14.2)             # question lands
    place(sfx, boom(0.45), 18.73)                         # FALCON wordmark reveal
    place(sfx, whoosh(0.5, True, 0.22), 20.5)             # zoom-through into shot 4
    place(sfx, tone(1320, 0.4, 0.08, 9), 22.0)            # target selected
    place(sfx, tone(1760, 0.35, 0.09, 10), 25.98)         # Messi: first real lock (source frame 62)
    place(sfx, tone(330, 0.5, 0.08, 6), 27.13)            # Messi: lock lost (source frame 91)
    for t in (32.75, 39.4, 44.75, 50.75):                          # whip transitions
        place(sfx, whoosh(0.55, True, 0.25), t)
    place(sfx, tone(1320, 0.4, 0.08, 9), 57.5)            # frisbee selected
    place(sfx, tone(330, 0.5, 0.08, 6), 62.37)            # frisbee: match lost at the catch
    place(sfx, boom(0.35), 67.0)                          # brand close
    for k, f in enumerate((1047, 1319)):                  # author reveal chime
        place(sfx, tone(f, 1.0, 0.07, 3), 73.33 + k * 0.4)
    out += sfx

    # ---- safety: soft limit to about -1 dBTP
    peak = np.max(np.abs(out))
    lim = 10 ** (-1.2 / 20)
    if peak > lim:
        out = np.tanh(out / lim) * lim
    i, tp = lufs(out[int(12 * SR):], work)
    i0, _ = lufs(out[:int(12 * SR)], work)
    print(f'intro {i0:.1f} LUFS, 12-80 s {i:.1f} LUFS, true peak {tp:.1f} dBTP')
    (work / 'soundtrack.f32').unlink(missing_ok=True)
    np.ascontiguousarray(out).tofile(work / 'soundtrack.f32')
    subprocess.check_call(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '2',
                           '-i', str(work / 'soundtrack.f32'), '-c:a', 'pcm_s16le', str(work / 'soundtrack.wav')])
    (work / '_meas.f32').unlink(missing_ok=True)
    print('wrote', work / 'soundtrack.wav')


if __name__ == '__main__':
    main()
