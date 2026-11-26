"""WER before/after LLM correction, key-decision recall, timing."""
import json
import random

import jiwer
import pandas as pd
from sentence_transformers import SentenceTransformer

from tracemeet.attribution import load_rows
from tracemeet.config import load_config
from tracemeet.evaluation.asr_metrics import normalize_gold, normalize_hyp
from tracemeet.pipeline import MEETINGS

EMB = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TAUS = (0.5, 0.6, 0.7)
LIMIT = 600.0

cfg = load_config()
root = cfg["paths"]["extraction_out"] / "pipeline"
out = cfg["paths"]["reports_out"]
out.mkdir(parents=True, exist_ok=True)


def counts(ref, hyp):
    n = len(ref.split())
    if n == 0:
        return None
    if not hyp.split():
        return {"err": n, "N": n}
    o = jiwer.process_words(ref, hyp)
    return {"err": o.substitutions + o.deletions + o.insertions, "N": n}


def track_text(rows, spk, norm):
    sel = sorted([r for r in rows if r["speaker"] == spk and r["start"] < LIMIT],
                 key=lambda r: (r["start"], r["end"]))
    return norm(" ".join(r["text"] for r in sel))


# ---------- 1. WER before / after correction ----------
gn = lambda t: normalize_gold(t, True)
hn = lambda t: normalize_hyp(t, True)
wer_rows, changed_examples = [], []
for mid in MEETINGS:
    gold = load_rows(cfg, mid, "gold")
    masked = load_rows(cfg, mid, "asr_masked")
    fixed = json.loads((root / "corrected" / f"{mid}.json").read_text())["rows"]
    for spk in sorted({r["speaker"] for r in gold}):
        ref = track_text(gold, spk, gn)
        b, a = counts(ref, track_text(masked, spk, hn)), counts(ref, track_text(fixed, spk, hn))
        if b and a:
            wer_rows.append({"meeting": mid, "speaker": spk, "N": b["N"],
                             "err_before": b["err"], "err_after": a["err"]})
    changed_examples += [(r["text_orig"], r["text"]) for r in fixed
                         if r.get("text_orig") and r["text"] != r["text_orig"]]
w = pd.DataFrame(wer_rows)
tab = w.groupby("meeting")[["N", "err_before", "err_after"]].sum()
tab.loc["ALL"] = tab.sum()
tab["WER_before_%"] = (100 * tab.err_before / tab.N).round(1)
tab["WER_after_%"] = (100 * tab.err_after / tab.N).round(1)
tab["delta_pts"] = (tab["WER_after_%"] - tab["WER_before_%"]).round(1)
print("== WER on the first 10 min (masked ASR), before vs after LLM correction ==")
print(tab[["N", "WER_before_%", "WER_after_%", "delta_pts"]].to_string())
tab.to_csv(out / "pipeline_wer_correction.csv")
random.seed(0)
print("\n== Examples of corrections (original -> corrected) ==")
for o, c in random.sample(changed_examples, min(6, len(changed_examples))):
    print(f"- {o}\n  -> {c}")

# ---------- 2. Key-decision / action recall (gold-derived reference) ----------
emb = SentenceTransformer(EMB)


def items(cond, mid, key):
    d = json.loads((root / "summaries" / cond / f"{mid}.json").read_text())
    res = []
    for wd in d["windows"]:
        if key == "decisions":
            res += wd.get("decisions", [])
        else:
            res += [f"{a.get('qui') or ''} {a['quoi']}".strip() for a in wd.get("actions", [])]
    return res


print("\n== Recall of key decisions/actions vs gold-derived reference (embedding match) ==")
rec = []
for key in ("decisions", "actions"):
    for cond in ("masked", "corrected"):
        for tau in TAUS:
            hit = tot = n_noisy = 0
            for mid in MEETINGS:
                g, n = items("gold", mid, key), items(cond, mid, key)
                n_noisy += len(n)
                if not g:
                    continue
                tot += len(g)
                if n:
                    sim = emb.encode(g, normalize_embeddings=True) @ emb.encode(n, normalize_embeddings=True).T
                    hit += int((sim.max(axis=1) >= tau).sum())
            rec.append({"type": key, "version": cond, "tau": tau, "gold_items": tot,
                        "noisy_items": n_noisy, "recall_%": round(100 * hit / tot, 1) if tot else None})
r = pd.DataFrame(rec)
print(r.to_string(index=False))
r.to_csv(out / "pipeline_key_recall.csv", index=False)

# ---------- 3. Timing ----------
print("\n== Processing time (seconds) ==")
trows = []
for mid in MEETINGS:
    base = cfg["paths"]["asr_out"] / "large-v3_mask3" / mid
    js = [json.loads(f.read_text()) for f in [base / "mix.json"] + sorted((base / "tracks").glob("*.json"))]
    row = {"meeting": mid, "asr_s(15min audio, mix+tracks)": round(sum(j["runtime_s"] for j in js), 1),
           "correction_s(10min)": json.loads((root / "corrected" / f"{mid}.json").read_text())["runtime_s"]}
    for cond in ("gold", "masked", "corrected"):
        row[f"minutes_{cond}_s"] = json.loads((root / "summaries" / cond / f"{mid}.json").read_text())["runtime_s"]
    trows.append(row)
t = pd.DataFrame(trows)
print(t.to_string(index=False))
t.to_csv(out / "pipeline_timing.csv", index=False)
