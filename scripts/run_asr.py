"""Transcribe selected meetings: mix.wav (baseline) and each speaker track.

--mask-margin X: for co-located recordings (meeting id place code H or L),
silence frames where another microphone dominates, before transcribing the
tracks (cross-talk suppression). Output goes to <model>_mask<X>/, which is
completed with copies of the unmasked run for everything that is not masked.
"""
import argparse
import json

from tracemeet.asr import load_audio, load_model, transcribe
from tracemeet.bleed import activity_mask, mask_audio
from tracemeet.config import load_config

COLOCATED = {"H", "L"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--language", default=None)
    ap.add_argument("--mask-margin", type=float, default=None)
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

    tag = f"_mask{args.mask_margin:g}" if args.mask_margin is not None else ""
    base_root = cfg["paths"]["asr_out"] / model_name.replace("/", "_")
    out_root = cfg["paths"]["asr_out"] / (model_name.replace("/", "_") + tag)
    raw = cfg["paths"]["raw"] / "summre"

    model = load_model(model_name)
    timing = {}
    for mid in meetings:
        out = out_root / mid
        (out / "tracks").mkdir(parents=True, exist_ok=True)
        use_mask = args.mask_margin is not None and mid[-1] in COLOCATED
        audio = None
        if use_mask:
            audio = {w.stem: load_audio(w) for w in (raw / mid / "tracks").glob("*.wav")}
            print(f"{mid}: co-located recording, masking with margin {args.mask_margin} dB")

        jobs = [("mix", raw / mid / "mix.wav", out / "mix.json")]
        jobs += [(f"track:{w.stem}", w, out / "tracks" / f"{w.stem}.json")
                 for w in sorted((raw / mid / "tracks").glob("*.wav"))]
        merged = []
        for name, wav, dest in jobs:
            masked_job = use_mask and name.startswith("track:")
            src = base_root / mid / dest.relative_to(out)
            if dest.exists() and not args.force:
                res = json.loads(dest.read_text())
            elif tag and not masked_job and src.exists() and not args.force:
                res = json.loads(src.read_text())  # unchanged: reuse the unmasked result
                dest.write_text(json.dumps(res, ensure_ascii=False, indent=1))
            else:
                a = None
                if masked_job:
                    spk = wav.stem
                    mask = activity_mask(audio, spk, args.mask_margin)
                    a = mask_audio(audio[spk], mask)
                    print(f"   {mid}/{spk}: kept {mask.mean() * 100:.0f}% of frames")
                res = transcribe(model, wav, language, audio=a)
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
