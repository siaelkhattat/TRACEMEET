"""Cross-talk (bleed) suppression for multi-microphone recordings."""
import numpy as np

SR = 16000


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9) if len(x) else -120.0


def gate_segments(segments, spk, tracks, margin_db=3.0):
    """Keep a segment only if the speaker's own mic is within margin_db of the loudest."""
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
