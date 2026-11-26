"""Per-meeting AUROC, leave-one-meeting-out AUROC, and flag-and-drop table."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tracemeet.config import load_config

cfg = load_config()
df = pd.read_csv(cfg["paths"]["reports_out"] / "attribution_claims.csv")
raw = df[(df.cond == "asr_raw") & df.label.isin(["correct", "misattributed"])].copy()
raw["y"] = (raw.label == "misattributed").astype(int)


def auc(g):
    pos = int(g.y.sum())
    if pos < 3 or len(g) - pos < 3:
        return None
    s = -g.dominance.fillna(g.dominance.median())
    return roc_auc_score(g.y, s)


print("== AUROC of low mic dominance, per meeting (asr_raw) ==")
for m, g in raw.groupby("meeting"):
    a = auc(g)
    print(f"{m}: n={len(g)} misattributed={int(g.y.sum())}  "
          + (f"AUROC {a:.2f}" if a is not None else "(too few of one class)"))

print("\n== Leave-one-meeting-out AUROC (asr_raw) ==")
for m in raw.meeting.unique():
    a = auc(raw[raw.meeting != m])
    print(f"without {m}: " + (f"AUROC {a:.2f}" if a is not None else "n/a"))

print("\n== Drop claims whose speaker's mic is not dominant (asr_raw, judged claims) ==")
print(f"no filter: coverage 100%, misattributed {raw.y.mean() * 100:.1f}%")
d = raw.dominance.fillna(raw.dominance.min())
lp = raw.avg_logprob.fillna(raw.avg_logprob.min())
for t in (-12, -9, -6, -3, 0):
    keep = d >= t
    cov = keep.mean()
    keep_lp = lp >= np.quantile(lp, 1 - cov)
    print(f"dominance >= {t:>3} dB | coverage {cov * 100:5.1f}% | misattributed "
          f"{raw.y[keep].mean() * 100:5.1f}% | logprob filter, same coverage: "
          f"{raw.y[keep_lp].mean() * 100:5.1f}%")
