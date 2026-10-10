# FALCON / Signal 13 — 80 s video (brief v3)

Output: `output/FALCON_Signal13_80s.mp4` (1920×1080, 30 fps, 2,400 frames, 80.000 s, AAC 48 kHz).
Burned-in captions plus `captions.srt`. Rebuild with `make_video.sh <work_dir>`.

## Storyboard (as rendered)

| # | Time | Content |
|---|---|---|
| 1 | 0:00–0:12 | `intro.mp4` 0–12 s, native commentary at −16 LUFS, 8-frame audio fade, "Original footage" label |
| 2 | 0:12–0:17 | Rewind transition back to intro 4.5 s; real ball (in flight, located by hand) circled; question typed in |
| 3 | 0:17–0:21 | "Yes. Only signals." → FALCON / Signal 13 wordmark, zoom-through |
| 4 | 0:21–0:33 | Messi: reference-image selection → genuine run → replay (2× zoom) → paused on accepted frame 78 |
| 5 | 0:33–0:45 | CONCEPTUAL sine/cosine waves; the crop is rebuilt from its waves with the repo's custom FFT; the peak is the **actual** correlation surface from Messi frame 78 |
| 6 | 0:45–0:51 | CONCEPTUAL Predict → Verify (Kalman prediction, PSR + NCC checks) |
| 7 | 0:51–0:57 | CONCEPTUAL global search grid, "Recovery is not guaranteed" |
| 8 | 0:57–1:07 | Frisbee: selection freeze, full 7.37 s clip at native speed, paused on the final real frame |
| 9 | 1:07–1:20 | Brand close, then author portraits from `Authors_pic/`; last 2 s still |

Changes from the brief, recorded in `storyboard.json`:
- Shot 4 runs 12 s, not 14 s, and shot 8 runs 10 s, not 14 s, because the frisbee clip is never slowed. The 6 s saved go to shot 9 for the author credits. The total is still 80 s.
- Shot 4 adds one honest line: "Here, it locks on for about a second, then loses it again."
- The 3 s template-adaptation beat from the 45 s cut is dropped for pacing, as the brief asks.

## What the tracker actually did (`evidence/*_track.json`)
- **Messi** (reference-image mode, 32×32 crop): SEARCHING until source frame 62. Then TRACKING for frames 62–90 (2.48–3.60 s, about 1.2 s), then SEARCHING to the end.
- **Frisbee** (manual ROI on frame 0): TRACKING for frames 0–124 (about 4.1 s), then lost near the catch and never re-acquired. The narration "until the catch makes it too hard" matches this.

## Audio
- Narration is generated locally with Kokoro-82M (voice `af_heart`), one take per caption phrase. The caption timings come straight from the take lengths.
- Higgsfield could not be used: the account has 0 credits. Microsoft Edge TTS is blocked by the network policy.
- Music bed (it ducks under the voice) and sound effects are synthesized in `audio_mix.py`. No third-party music is used.
- Measured loudness: intro −16.3 LUFS, 12–80 s −16.6 LUFS, true peak −2.7 dBTP.

## Copyright note
Broadcast footage with commentary may be muted or blocked on Facebook. Check before uploading. If needed, a muted-commentary fallback can be made by dropping the intro audio in `audio_mix.py`.
