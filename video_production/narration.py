"""Synthesize narration with Kokoro-82M (local, offline): one WAV per caption phrase.

Usage: python video_production/narration.py <model_dir> <out_dir>
Writes <out_dir>/<shot>_<n>.wav and <out_dir>/takes.json (phrase durations).
"""
import json
import sys
from pathlib import Path

import soundfile as sf
from kokoro_onnx import Kokoro

VOICE = 'af_heart'
SPEED = 1.0

if __name__ == '__main__':
    sb = json.loads(Path(__file__).with_name('storyboard.json').read_text())
    model_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    tts = Kokoro(str(model_dir / 'kokoro-v1.0.onnx'), str(model_dir / 'voices-v1.0.bin'))
    takes = {}
    for take in sb['narration']:
        rows = []
        for n, phrase in enumerate(take['phrases']):
            audio, sr = tts.create(phrase, voice=VOICE, speed=SPEED, lang='en-us')
            path = out_dir / f"{take['shot']}_{n}.wav"
            sf.write(path, audio, sr)
            rows.append({'text': phrase, 'wav': path.name, 'dur': round(len(audio) / sr, 3), 'sr': sr})
        takes[take['shot']] = rows
        print(take['shot'], [r['dur'] for r in rows], round(sum(r['dur'] for r in rows), 2))
    (out_dir / 'takes.json').write_text(json.dumps({'voice': VOICE, 'speed': SPEED, 'takes': takes}, indent=1))
