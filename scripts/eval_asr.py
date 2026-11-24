"""WER/CER vs gold: per speaker track (System 2 input) and on the mix (baseline)."""
import argparse
import json

import jiwer
import pandas as pd

from tracemeet.asr import load_audio
from tracemeet.bleed import gate_segments
from tracemeet.config import load_config
from tracemeet.evaluation.asr_metrics import normalize_gold, normalize_hyp


def counts(ref, hyp, level):
    if not ref.split():
        return None
    if not hyp.split():
        n = len(ref.split()) if level == "word" else len(ref)
        return {"S": 0, "D": n, "I": 0, "N": n}
    f = jiwer.process_words if level == "word" else jiwer.process_characters
    o = f(ref, hyp)
    return {"S": o.substitutions, "D": o.deletions, "I": o.insertions,
            "N": o.substitutions + o.deletions + o.hits}


def seg_text(segs, key):
    return " ".join(s[key] for s in sorted(segs, key=lambda s: (s["start"], s["end"])))


def rate(rows, level):
    tot = {k: sum(r[level][k] for r in rows if r[level]) for k in "SDIN"}
    return round(100 * (tot["S"] + tot["D"] + tot["I"]) / tot["N"], 1) if tot["N"] else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--gate", type=float, default=None, help="bleed gating margin in dB")
    args = ap.parse_args()
    tag = f"_gate{args.gate:g}" if args.gate is not None else ""
    cfg = load_config()
    model = args.model or cfg["asr"]["model"]
    sel = json.loads((cfg["paths"]["splits"] / "selected_meetings.json").read_text())
    splits = ["dev", "test"] if args.split == "all" else [args.split]
    meetings = [m for s in splits for m in sel[s]]
    asr_root = cfg["paths"]["asr_out"] / model.replace("/", "_")
    raw = cfg["paths"]["raw"] / "summre"

    rows = []
    for mid in meetings:
        gold = json.loads((raw / mid / "gold_segments.json").read_text())
        hyp_mix = json.loads((asr_root / mid / "mix.json").read_text())["segments"]
        all_gold = [s for segs in gold.values() for s in segs]
        ref = normalize_gold(seg_text(all_gold, "transcript"))
        hyp = normalize_hyp(seg_text(hyp_mix, "text"))
        rows.append({"meeting": mid, "kind": "mix", "unit": "mix",
                     "word": counts(ref, hyp, "word"), "char": counts(ref, hyp, "char")})
        tracks_audio = None
        if args.gate is not None:
            tracks_audio = {k: load_audio(raw / mid / "tracks" / f"{k}.wav") for k in gold}
        for spk, segs in gold.items():
            hp = asr_root / mid / "tracks" / f"{spk}.json"
            h = json.loads(hp.read_text())["segments"]
            if tracks_audio is not None:
                h = gate_segments(h, spk, tracks_audio, args.gate)
            ref = normalize_gold(seg_text(segs, "transcript"))
            hyp = normalize_hyp(seg_text(h, "text"))
            rows.append({"meeting": mid, "kind": "tracks", "unit": spk,
                         "word": counts(ref, hyp, "word"), "char": counts(ref, hyp, "char")})

    table = []
    for mid in meetings + ["ALL"]:
        sub = [r for r in rows if mid == "ALL" or r["meeting"] == mid]
        mix = [r for r in sub if r["kind"] == "mix"]
        trk = [r for r in sub if r["kind"] == "tracks"]
        table.append({"meeting": mid,
                      "WER_mix": rate(mix, "word"), "CER_mix": rate(mix, "char"),
                      "WER_tracks": rate(trk, "word"), "CER_tracks": rate(trk, "char")})
    df = pd.DataFrame(table)
    detail = pd.DataFrame([{"meeting": r["meeting"], "kind": r["kind"], "unit": r["unit"],
                            "WER": rate([r], "word"), "CER": rate([r], "char")} for r in rows])
    out = cfg["paths"]["reports_out"]
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"asr_summary_{model}_{args.split}{tag}.csv", index=False)
    detail.to_csv(out / f"asr_detail_{model}_{args.split}{tag}.csv", index=False)
    print(f"\nModel: {model} | split: {args.split}  (pooled, %)")
    print(df.to_string(index=False))
    print("\nPer track:")
    print(detail[detail.kind == "tracks"].to_string(index=False))


if __name__ == "__main__":
    main()
