"""Transcribe selected meetings: mix.wav (baseline) and each speaker track."""
import argparse
import json

from tracemeet.asr import load_model, transcribe
from tracemeet.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--language", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    model_name = args.model or cfg["asr"]["model"]
    language = args.language or cfg["asr"]["language"]
    if language == "auto":
        language = None
    sel = json.loads((cfg["paths"]["splits"] / "selected_meetings.json").read_text())
    splits = ["dev", "test"] if args.split == "all" else [args.split]
    meetings = [m for s in splits for m in sel[s]]

    model = load_model(model_name)
    raw = cfg["paths"]["raw"] / "summre"
    out_root = cfg["paths"]["asr_out"] / model_name.replace("/", "_")
    timing = {}
    for mid in meetings:
        out = out_root / mid
        (out / "tracks").mkdir(parents=True, exist_ok=True)
        jobs = [("mix", raw / mid / "mix.wav", out / "mix.json")]
        jobs += [(f"track:{w.stem}", w, out / "tracks" / f"{w.stem}.json")
                 for w in sorted((raw / mid / "tracks").glob("*.wav"))]
        merged = []
        for name, wav, dest in jobs:
            if dest.exists() and not args.force:
                res = json.loads(dest.read_text())
            else:
                res = transcribe(model, wav, language)
                dest.write_text(json.dumps(res, ensure_ascii=False, indent=1))
                print(f"{mid} {name}: {res['duration']:.0f}s audio in {res['runtime_s']}s")
            timing[f"{mid}/{name}"] = {"audio_s": res["duration"], "runtime_s": res["runtime_s"]}
            if name.startswith("track:"):
                spk = name.split(":", 1)[1]
                merged += [{**s, "speaker": spk} for s in res["segments"]]
        merged.sort(key=lambda s: (s["start"], s["end"]))
        (out / "merged.json").write_text(json.dumps(merged, ensure_ascii=False, indent=1))
    (out_root / "timing.json").write_text(json.dumps(timing, indent=1))
    print("done ->", out_root)


if __name__ == "__main__":
    main()
