"""Label claims as correct / misattributed / unverifiable against the gold transcript,
then test whether mic dominance flags the misattributed ones."""
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score

from tracemeet.asr import load_audio
from tracemeet.attribution import CONDS, load_rows
from tracemeet.bleed import SR, rms_db
from tracemeet.config import load_config

EMB = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
WIN, MIN_SIM, GAP, DELTA = 40.0, 0.30, 2.0, 0.05
COLOCATED = {"H", "L"}


def gold_chunks(rows):
    chunks, last = [], {}
    for r in rows:
        c = last.get(r["speaker"])
        if c and r["start"] - c["end"] < GAP:
            c["end"] = max(c["end"], r["end"])
            c["text"] += " " + r["text"]
        else:
            c = {"speaker": r["speaker"], "start": r["start"], "end": r["end"], "text": r["text"]}
            chunks.append(c)
            last[r["speaker"]] = c
    return [c for c in chunks if len(c["text"].split()) >= 4]


def lab_from(top, own, delta):
    if top < MIN_SIM:
        return "unverifiable"
    return "correct" if own >= top - delta else "misattributed"


def features(claim, rows, tracks):
    spk, t = claim["speaker"], claim["time"]
    cand = [r for r in rows if r["speaker"] == spk and abs(r["start"] - t) <= 15]
    if cand:
        r = min(cand, key=lambda r: abs(r["start"] - t))
        a, b = r["start"], max(r["end"], r["start"] + 1.0)
        lp, ns = r.get("avg_logprob", np.nan), r.get("no_speech_prob", np.nan)
    else:
        a, b, lp, ns = t, t + 4.0, np.nan, np.nan
    dom = np.nan
    if spk in tracks and len(tracks) > 1:
        ia, ib = int(a * SR), int(b * SR)
        own = rms_db(tracks[spk][ia:ib])
        dom = own - max(rms_db(x[ia:ib]) for k, x in tracks.items() if k != spk)
    return {"dominance": dom, "avg_logprob": lp, "no_speech_prob": ns}


def auroc_ci(y, s, n=1000, seed=0):
    y, s = np.asarray(y, int), np.asarray(s, float)
    if np.isnan(s).all() or y.sum() < 3 or (1 - y).sum() < 3:
        return None
    s = np.where(np.isnan(s), np.nanmedian(s), s)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], s[i]))
    return roc_auc_score(y, s), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def main():
    cfg = load_config()
    sel = json.loads((cfg["paths"]["splits"] / "selected_meetings.json").read_text())
    meetings = [m for s in ("dev", "test") for m in sel[s]]
    emb = SentenceTransformer(EMB)
    recs = []
    for mid in meetings:
        chunks = gold_chunks(load_rows(cfg, mid, "gold"))
        cemb = emb.encode([c["text"] for c in chunks], normalize_embeddings=True)
        raw = cfg["paths"]["raw"] / "summre" / mid / "tracks"
        tracks = {p.stem: load_audio(p) for p in raw.glob("*.wav")}
        for cond in CONDS:
            f = cfg["paths"]["extraction_out"] / "attribution" / cond / f"{mid}.json"
            if not f.exists():
                print("!! missing", mid, cond)
                continue
            claims = json.loads(f.read_text())["claims"]
            if not claims:
                continue
            rows = load_rows(cfg, mid, cond) if cond != "gold" else None
            qv = emb.encode([c["claim"] for c in claims], normalize_embeddings=True)
            for c, q in zip(claims, qv):
                idx = [i for i, ch in enumerate(chunks)
                       if ch["end"] >= c["time"] - WIN and ch["start"] <= c["time"] + WIN]
                best = {}
                for i, s in zip(idx, cemb[idx] @ q):
                    best[chunks[i]["speaker"]] = max(best.get(chunks[i]["speaker"], -1.0), float(s))
                if best:
                    top_sp = max(best, key=best.get)
                    top, own = best[top_sp], best.get(c["speaker"], -1.0)
                else:
                    top_sp, top, own = None, -1.0, -1.0
                rec = {"meeting": mid, "colocated": mid[-1] in COLOCATED, "cond": cond,
                       "speaker": c["speaker"], "time": c["time"], "claim": c["claim"],
                       "top_speaker": top_sp, "top_sim": top, "own_sim": own}
                rec.update(features(c, rows, tracks) if cond != "gold" else
                           {"dominance": np.nan, "avg_logprob": np.nan, "no_speech_prob": np.nan})
                recs.append(rec)
    df = pd.DataFrame(recs)
    out = cfg["paths"]["reports_out"]
    out.mkdir(parents=True, exist_ok=True)
    df["label"] = [lab_from(t, o, DELTA) for t, o in zip(df.top_sim, df.own_sim)]
    df.to_csv(out / "attribution_claims.csv", index=False)

    print("\n== Claim labels per transcript version (delta=0.05, %) ==")
    tab = pd.crosstab(df.cond, df.label, normalize="index").mul(100).round(1)
    tab.insert(0, "n_claims", df.groupby("cond").size())
    print(tab.to_string())
    print("\n== Misattributed % by version x recording type (colocated=studio) ==")
    print(df.assign(m=df.label == "misattributed")
            .groupby(["cond", "colocated"]).m.mean().mul(100).round(1).to_string())
    print("\n== Sensitivity of misattributed % to delta ==")
    for d in (0.0, 0.05, 0.10):
        m = [lab_from(t, o, d) == "misattributed" for t, o in zip(df.top_sim, df.own_sim)]
        print(f"delta={d}:", df.assign(m=m).groupby("cond").m.mean().mul(100).round(1).to_dict())

    print("\n== Can a signal flag misattributed claims? AUROC (95% CI), claims judged only ==")
    for name, conds in (("asr_raw", ["asr_raw"]), ("asr_raw + asr_masked", ["asr_raw", "asr_masked"])):
        sub = df[df.cond.isin(conds) & df.label.isin(["correct", "misattributed"])]
        y = (sub.label == "misattributed").astype(int)
        print(f"\n[{name}] n={len(sub)}, misattributed={int(y.sum())}")
        for feat, s in (("low mic dominance", -sub.dominance), ("low ASR logprob", -sub.avg_logprob),
                        ("high no-speech prob", sub.no_speech_prob)):
            r = auroc_ci(y, s)
            print(f"  {feat:22s}", "too few positives or no data" if r is None
                  else f"AUROC {r[0]:.2f} ({r[1]:.2f}-{r[2]:.2f})")

    ex = df[(df.cond == "asr_raw") & (df.label == "misattributed")].head(5)
    print("\n== Examples (asr_raw, misattributed) ==")
    for _, r in ex.iterrows():
        print(f"- claimed S{r.speaker}, closer to S{r.top_speaker}, dominance {r.dominance:.1f} dB: {r.claim}")


if __name__ == "__main__":
    main()
