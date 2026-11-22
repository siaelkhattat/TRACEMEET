"""Download the selected SUMM-RE meetings (run on Colab, CPU is enough).

Per meeting, under <DATA_ROOT>/data/raw/summre/<meeting_id>/:
  tracks/<speaker>.wav   16 kHz mono, cropped to the excerpt window
  segments/<speaker>.json gold segments (timestamps relative to window start)
  mix.wav                mixed-down audio (baseline input, no speaker info)
  gold_segments.json     all speakers merged
  meta.json
Only the parquet files that contain the selected meetings are downloaded,
one at a time, and deleted after use.
"""
import argparse
import io
import json
import math
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfFileSystem, hf_hub_download
from scipy.signal import resample_poly

from tracemeet.config import load_config

REPO_ID = "linagora/SUMM-RE"
FS_ROOT = f"datasets/{REPO_ID}/"
TARGET_SR = 16000
TMP = Path("/tmp/summre_tmp")


def split_of(path):
    name = path.lower().rsplit("/", 1)[-1]
    for s in ("train", "dev", "test"):
        if name.startswith(s):
            return s
    return None


def load_crop(audio_bytes, start_s, end_s):
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    data = data.mean(axis=1)[int(start_s * sr):int(end_s * sr)]
    if sr != TARGET_SR:
        g = math.gcd(sr, TARGET_SR)
        data = resample_poly(data, TARGET_SR // g, sr // g).astype("float32")
    return data


def clip_segments(segments, start_s, end_s):
    out = []
    for s in segments:
        if s["end"] <= start_s or s["start"] >= end_s:
            continue
        words = [{**w, "start": w["start"] - start_s, "end": w["end"] - start_s}
                 for w in (s.get("words") or [])
                 if w["end"] > start_s and w["start"] < end_s]
        text = s["transcript"]
        if s["start"] < start_s or s["end"] > end_s:  # segment cut by the window
            text = " ".join(w["word"] for w in words)
        out.append({"start": max(s["start"], start_s) - start_s,
                    "end": min(s["end"], end_s) - start_s,
                    "transcript": text, "words": words})
    return out


def process_file(rel, wanted, splits, out_root, start_s, end_s):
    print(f"\n== downloading {rel}")
    local = hf_hub_download(REPO_ID, rel, repo_type="dataset", local_dir=TMP)
    pf = pq.ParquetFile(local, memory_map=True)
    print(f"   row groups: {pf.metadata.num_row_groups}, rows: {pf.metadata.num_rows}")
    cols = ["meeting_id", "speaker_id", "audio", "segments"]
    for batch in pf.iter_batches(batch_size=1, columns=cols):
        row = batch.to_pylist()[0]
        mid, spk = row["meeting_id"], str(row["speaker_id"])
        if mid not in wanted:
            continue
        mdir = out_root / mid
        (mdir / "tracks").mkdir(parents=True, exist_ok=True)
        (mdir / "segments").mkdir(exist_ok=True)
        wav_p, seg_p = mdir / "tracks" / f"{spk}.wav", mdir / "segments" / f"{spk}.json"
        if wav_p.exists() and seg_p.exists():
            print(f"   skip {mid}/{spk} (already done)")
            continue
        audio = row["audio"]["bytes"]
        sf.write(wav_p, load_crop(audio, start_s, end_s), TARGET_SR, subtype="PCM_16")
        seg_p.write_text(json.dumps(clip_segments(row["segments"], start_s, end_s),
                                    ensure_ascii=False))
        print(f"   saved {mid}/{spk}")
    del pf
    os.remove(local)


def finalize(mdir, mid, split, start_s, end_s):
    tracks = sorted((mdir / "tracks").glob("*.wav"))
    if not tracks:
        print(f"!! no tracks found for {mid}")
        return
    arrays = [sf.read(t, dtype="float32")[0] for t in tracks]
    mix = np.zeros(max(len(a) for a in arrays), dtype="float32")
    for a in arrays:
        mix[:len(a)] += a
    mix = mix / (float(np.abs(mix).max()) or 1.0) * 0.95
    sf.write(mdir / "mix.wav", mix, TARGET_SR, subtype="PCM_16")
    gold = {t.stem: json.loads((mdir / "segments" / f"{t.stem}.json").read_text())
            for t in tracks}
    (mdir / "gold_segments.json").write_text(json.dumps(gold, ensure_ascii=False, indent=1))
    (mdir / "meta.json").write_text(json.dumps({
        "meeting_id": mid, "split": split, "source": REPO_ID,
        "speakers": [t.stem for t in tracks], "sr": TARGET_SR,
        "window_start_s": start_s, "window_end_s": end_s,
        "duration_s": len(mix) / TARGET_SR}, indent=2))
    print(f"finalized {mid}: {len(tracks)} speakers, {len(mix) / TARGET_SR / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--start-min", type=float, default=0.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    minutes = args.minutes or cfg["data"]["excerpt_minutes"]
    start_s, end_s = args.start_min * 60, (args.start_min + minutes) * 60
    sel = json.loads((cfg["paths"]["splits"] / "selected_meetings.json").read_text())
    meeting_split = {m: s for s, ms in sel.items() for m in ms}
    out_root = cfg["paths"]["raw"] / "summre"
    todo = {m for m in meeting_split
            if args.force or not (out_root / m / "meta.json").exists()}
    print(f"window {start_s / 60:.0f}-{end_s / 60:.0f} min | to fetch: {sorted(todo)}")
    if not todo:
        print("nothing to do")
        return

    fs = HfFileSystem()
    splits = {meeting_split[m] for m in todo}
    files = [f for f in fs.glob(f"{FS_ROOT}**/*.parquet")
             if split_of(f) in splits and "example" not in f.lower()]
    file_map = {}
    for f in files:
        ids = set(pq.read_table(f, columns=["meeting_id"], filesystem=fs)
                  .column("meeting_id").to_pylist())
        if ids & todo:
            file_map[f] = ids & todo
    print(f"{len(file_map)} parquet files contain the selected meetings")

    TMP.mkdir(parents=True, exist_ok=True)
    for f, wanted in sorted(file_map.items()):
        process_file(f[len(FS_ROOT):], wanted, splits, out_root, start_s, end_s)

    for m in sorted(todo):
        finalize(out_root / m, m, meeting_split[m], start_s, end_s)


if __name__ == "__main__":
    main()
