"""Why is a track's WER high? Error types, frequent errors, bleed estimate."""
import argparse
import json
from collections import Counter

import jiwer

from tracemeet.asr import load_audio
from tracemeet.bleed import gate_segments
from tracemeet.config import load_config
from tracemeet.evaluation.asr_metrics import normalize_gold, normalize_hyp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--meeting", required=True)
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--margin", type=float, default=3.0)
    a = ap.parse_args()
    cfg = load_config()
    raw = cfg["paths"]["raw"] / "summre" / a.meeting
    gold = json.loads((raw / "gold_segments.json").read_text())
    hyp_segs = json.loads((cfg["paths"]["asr_out"] / a.model.replace("/", "_") / a.meeting
                           / "tracks" / f"{a.speaker}.json").read_text())["segments"]

    ref = normalize_gold(" ".join(s["transcript"] for s in gold[a.speaker]))
    hyp = normalize_hyp(" ".join(s["text"] for s in hyp_segs))
    o = jiwer.process_words(ref, hyp)
    print(f"ref words {len(ref.split())} | hyp words {len(hyp.split())} | "
          f"S {o.substitutions} D {o.deletions} I {o.insertions} | WER {o.wer * 100:.1f}%")

    ref_w, hyp_w = o.references[0], o.hypotheses[0]
    dele, ins = Counter(), Counter()
    for c in o.alignments[0]:
        if c.type == "delete":
            dele.update(ref_w[c.ref_start_idx:c.ref_end_idx])
        elif c.type == "insert":
            ins.update(hyp_w[c.hyp_start_idx:c.hyp_end_idx])
    print("top deleted :", dele.most_common(15))
    print("top inserted:", ins.most_common(15))

    tracks = {p.stem: load_audio(p) for p in (raw / "tracks").glob("*.wav")}
    kept = gate_segments(hyp_segs, a.speaker, tracks, a.margin)
    w_all = sum(len(s["text"].split()) for s in hyp_segs)
    w_kept = sum(len(s["text"].split()) for s in kept)
    print(f"\nbleed check (margin {a.margin} dB): {len(hyp_segs) - len(kept)}/{len(hyp_segs)} "
          f"segments, {w_all - w_kept}/{w_all} hyp words are louder on another mic")
    print("\nREF:", ref[:400])
    print("HYP:", hyp[:400])


if __name__ == "__main__":
    main()
