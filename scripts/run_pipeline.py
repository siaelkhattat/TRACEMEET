"""Batch: LLM correction of masked ASR, then minutes (summary + decisions + actions)
from gold / masked ASR / corrected ASR. First 10 min of each meeting. Resumable."""
import argparse
import json
import time

import torch

from tracemeet.attribution import load_rows
from tracemeet.config import load_config
from tracemeet.llm import load_llm
from tracemeet.pipeline import MEETINGS, WINDOW_S, correct_window, make_windows, summarize_window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=float, default=600.0, help="seconds processed per meeting")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    root = cfg["paths"]["extraction_out"] / "pipeline"
    tok, model = load_llm(cfg["llm"]["model"])
    shown = {"fix": False, "sum": False}

    for mid in MEETINGS:
        masked = [r for r in load_rows(cfg, mid, "asr_masked") if r["start"] < args.limit]

        fix_p = root / "corrected" / f"{mid}.json"
        if fix_p.exists() and not args.force:
            corrected = json.loads(fix_p.read_text())["rows"]
            print("skip correction", mid)
        else:
            t0, corrected, rej, failed = time.time(), [], 0, 0
            for w in make_windows(masked, WINDOW_S, args.limit):
                try:
                    rows, r, raw = correct_window(tok, model, w)
                    rej += r
                    if not shown["fix"]:
                        print("--- sample correction output ---\n" + raw[:700] + "\n---")
                        shown["fix"] = True
                except Exception as e:
                    print("!! correction failed on a window:", type(e).__name__, str(e)[:150])
                    rows, failed = w, failed + 1
                    torch.cuda.empty_cache()
                corrected += rows
            fix_p.parent.mkdir(parents=True, exist_ok=True)
            fix_p.write_text(json.dumps({"meeting": mid, "limit_s": args.limit,
                                         "runtime_s": round(time.time() - t0, 1),
                                         "n_rejected": rej, "n_failed_windows": failed,
                                         "rows": corrected}, ensure_ascii=False, indent=1))
            ch = sum(r["text"] != r.get("text_orig", r["text"]) for r in corrected)
            print(f"{mid} correction: {ch}/{len(corrected)} rows changed, {rej} rejected, "
                  f"{failed} failed windows, {time.time() - t0:.0f}s")

        sets = {"gold": [r for r in load_rows(cfg, mid, "gold") if r["start"] < args.limit],
                "masked": masked, "corrected": corrected}
        for cond, rows in sets.items():
            dest = root / "summaries" / cond / f"{mid}.json"
            if dest.exists() and not args.force:
                print("skip", mid, cond)
                continue
            t0, wins = time.time(), []
            for w in make_windows(rows, WINDOW_S, args.limit):
                idx = int(w[0]["start"] // WINDOW_S)
                try:
                    res = summarize_window(tok, model, w)
                    if not shown["sum"]:
                        print("--- sample minutes output ---\n" + res["raw"][:700] + "\n---")
                        shown["sum"] = True
                except Exception as e:
                    print("!! summary failed on a window:", type(e).__name__, str(e)[:150])
                    res = {"error": str(e)[:150]}
                    torch.cuda.empty_cache()
                wins.append({"window": idx, "start_s": idx * WINDOW_S, **res})
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps({"meeting": mid, "cond": cond,
                                        "runtime_s": round(time.time() - t0, 1),
                                        "windows": wins}, ensure_ascii=False, indent=1))
            nd = sum(len(w.get("decisions", [])) for w in wins)
            na = sum(len(w.get("actions", [])) for w in wins)
            print(f"{mid} {cond}: {nd} decisions, {na} actions, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
