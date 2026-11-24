"""Cross-talk (bleed) handling for multi-microphone recordings."""
import numpy as np

SR = 16000
HOP = 320  # 20 ms frames


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9) if len(x) else -120.0


def gate_segments(segments, spk, tracks, margin_db=3.0):
    """Segment-level gating (kept for the diagnostics script)."""
    keep = []
    for s in segments:
        a, b = int(s["start"] * SR), int(s["end"] * SR)
        if b <= a:
            continue
        own = rms_db(tracks[spk][a:b])
        best = max(rms_db(t[a:b]) for t in tracks.values())
        if own >= best - margin_db:
            keep.append(s)
    return keep


def frame_levels_db(x, hop=HOP):
    n = len(x) // hop
    if n == 0:
        return np.zeros(0)
    fr = x[: n * hop].reshape(n, hop)
    return 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-9)


def _smooth(v, k):
    if k <= 1 or len(v) == 0:
        return v
    pad = k // 2
    vp = np.pad(v, (pad, pad), mode="edge")
    return np.convolve(vp, np.ones(k) / k, mode="valid")[: len(v)]


def activity_mask(tracks, spk, margin_db=3.0, smooth_s=0.3, floor_margin_db=6.0, pad_s=0.3):
    """Boolean frame mask: True where `spk`'s own mic dominates the other mics."""
    k = int(smooth_s * SR / HOP) | 1
    levels = {n: _smooth(frame_levels_db(t), k) for n, t in tracks.items()}
    own = levels[spk]
    others = [v for n, v in levels.items() if n != spk]
    m = min([len(own)] + [len(v) for v in others])
    own = own[:m]
    if others:
        best_other = np.max(np.stack([v[:m] for v in others]), axis=0)
        dominant = own >= best_other - margin_db
    else:
        dominant = np.ones(m, dtype=bool)
    floor = np.percentile(own, 10)
    mask = dominant & (own > floor + floor_margin_db)
    w = int(pad_s * SR / HOP) * 2 + 1  # dilate so word edges are not clipped
    return np.convolve(mask.astype(float), np.ones(w), mode="same") > 0


def mask_audio(audio, mask):
    keep = np.repeat(mask, HOP)
    keep = np.pad(keep, (0, max(0, len(audio) - len(keep))), constant_values=False)[: len(audio)]
    out = audio.copy()
    out[~keep] = 0.0
    return out
