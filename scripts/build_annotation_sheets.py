"""Readable gold transcript + blank annotation file per selected meeting."""
import json

from tracemeet.config import load_config
from tracemeet.evaluation.asr_metrics import strip_gold_markup


def mmss(t):
    m, s = divmod(int(t), 60)
    return f"{m:02d}:{s:02d}"


def main():
    cfg = load_config()
    raw = cfg["paths"]["raw"] / "summre"
    out_root = cfg["paths"]["annotations"]
    sel = json.loads((cfg["paths"]["splits"] / "selected_meetings.json").read_text())
    for split, meetings in sel.items():
        for mid in meetings:
            gold = json.loads((raw / mid / "gold_segments.json").read_text())
            meta = json.loads((raw / mid / "meta.json").read_text())
            rows = []
            for spk, segs in gold.items():
                for s in segs:
                    text = " ".join(strip_gold_markup(s["transcript"]).split())
                    if text:
                        rows.append((s["start"], s["end"], spk, text))
            rows.sort()
            d = out_root / mid
            d.mkdir(parents=True, exist_ok=True)
            lines = [f"# {mid} ({split}): gold transcript, window 00:00-{mmss(meta['duration_s'])}",
                     "Format: [mm:ss | start-end seconds] **Sspeaker**: text", ""]
            lines += [f"[{mmss(a)} | {a:.1f}-{b:.1f}] **S{spk}**: {t}" for a, b, spk, t in rows]
            (d / "transcript.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            ann = d / "annotations.json"
            if not ann.exists():
                ann.write_text(json.dumps({
                    "meeting_id": mid, "split": split, "annotator": "",
                    "window_s": [0, meta["duration_s"]], "topic": "", "items": []},
                    ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{mid}: {len(rows)} utterances -> {d}")


if __name__ == "__main__":
    main()
