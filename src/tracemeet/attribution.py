"""Helpers for the 'the microphone knows who spoke' experiment."""
import json
import re

from tracemeet.evaluation.asr_metrics import strip_gold_markup

CONDS = {"gold": None, "asr_masked": "large-v3_mask3", "asr_raw": "large-v3"}


def load_rows(cfg, mid, cond):
    raw = cfg["paths"]["raw"] / "summre" / mid
    rows = []
    if cond == "gold":
        gold = json.loads((raw / "gold_segments.json").read_text())
        for spk, segs in gold.items():
            for s in segs:
                t = " ".join(strip_gold_markup(s["transcript"]).split())
                if t:
                    rows.append({"speaker": spk, "start": s["start"], "end": s["end"], "text": t})
    else:
        root = cfg["paths"]["asr_out"] / CONDS[cond] / mid / "tracks"
        for p in sorted(root.glob("*.json")):
            for s in json.loads(p.read_text())["segments"]:
                rows.append({"speaker": p.stem, "start": s["start"], "end": s["end"],
                             "text": s["text"], "avg_logprob": s["avg_logprob"],
                             "no_speech_prob": s["no_speech_prob"]})
    rows.sort(key=lambda r: (r["start"], r["end"]))
    return rows


def format_transcript(rows):
    return "\n".join(f"[{int(r['start'])}] S{r['speaker']}: {r['text']}"
                     for r in rows if r["text"].strip())


def parse_claims(text):
    m = re.search(r"\[.*\]", text, flags=re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group())
    except json.JSONDecodeError:
        return []
    out = []
    for it in items:
        try:
            out.append({"speaker": re.sub(r"^S", "", str(it["speaker"]).strip()),
                        "time": float(it["time"]), "claim": str(it["claim"]).strip()})
        except (KeyError, TypeError, ValueError):
            continue
    return out
