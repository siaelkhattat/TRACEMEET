"""Select SUMM-RE meetings from metadata only (no audio is downloaded).

Rule: real experiments (E), decision meetings (D), Zoom or studio (Z/H),
3-4 speakers, varied scenarios. 2 meetings from dev (tuning), 3 from test.
"""
import json
import re

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from tracemeet.config import load_config

REPO = "datasets/linagora/SUMM-RE"
ID_RE = r"(\d{3})([a-z])_([EP])([A-E])([RDP])([LHZD])"
N_DEV, N_TEST = 2, 3


def split_of(path):
    p = path.lower()
    name = p.rsplit("/", 1)[-1]
    for s in ("train", "dev", "test"):
        if name.startswith(s) or f"/{s}/" in p:
            return s
    return None


def pick(sub, n):
    sub = sub.sort_values("meeting_id")
    chosen, seen = [], set()
    for _, r in sub.iterrows():  # first pass: one meeting per scenario
        if r.scenario not in seen:
            chosen.append(r)
            seen.add(r.scenario)
        if len(chosen) == n:
            return chosen
    for _, r in sub.iterrows():  # second pass: fill up if scenarios run out
        if len(chosen) == n:
            break
        if r.meeting_id not in [c.meeting_id for c in chosen]:
            chosen.append(r)
    return chosen


def main():
    fs = HfFileSystem()
    files = [f for f in fs.glob(f"{REPO}/**/*.parquet")
             if split_of(f) in ("dev", "test") and "example" not in f.lower()]
    print(f"{len(files)} parquet files for dev/test")
    frames = []
    for f in files:
        t = pq.read_table(f, columns=["meeting_id", "speaker_id"], filesystem=fs).to_pandas()
        t["split"] = split_of(f)
        frames.append(t)
        print("  read", f.rsplit("/", 1)[-1], len(t), "tracks")
    df = pd.concat(frames).drop_duplicates()
    print(df.head())

    meta = (df.groupby(["split", "meeting_id"]).speaker_id.nunique()
              .rename("n_speakers").reset_index())
    parsed = meta.meeting_id.str.extract(ID_RE)
    parsed.columns = ["exp", "order", "kind", "scenario", "mtype", "place"]
    meta = pd.concat([meta, parsed], axis=1)

    out = load_config()["paths"]["splits"]
    out.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out / "summre_candidates.csv", index=False)

    cand = meta[(meta.kind == "E") & (meta.mtype == "D")
                & (meta.place.isin(["Z", "H"]))
                & (meta.n_speakers.between(3, 4))]
    print(f"\n{len(cand)} candidate decision meetings (Z/H, 3-4 speakers)")

    sel = {
        "dev": [r.meeting_id for r in pick(cand[cand.split == "dev"], N_DEV)],
        "test": [r.meeting_id for r in pick(cand[cand.split == "test"], N_TEST)],
    }
    (out / "selected_meetings.json").write_text(json.dumps(sel, indent=2))
    print("\nSELECTED:")
    print(json.dumps(sel, indent=2))
    print(cand[cand.meeting_id.isin(sel["dev"] + sel["test"])]
          [["split", "meeting_id", "n_speakers", "scenario", "place"]].to_string(index=False))


if __name__ == "__main__":
    main()
