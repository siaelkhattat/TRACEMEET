"""Check annotation files for schema and consistency errors."""
import json
import sys

from tracemeet.config import load_config

TYPES = {"decision", "action", "rejected_proposal"}
LANGS = {"fr", "darija", "mixed"}


def check(path):
    errs = []
    a = json.loads(path.read_text(encoding="utf-8"))
    lo, hi = a["window_s"]
    for i, it in enumerate(a["items"]):
        p = f"item {i}"
        if it.get("type") not in TYPES:
            errs.append(f"{p}: type must be one of {sorted(TYPES)}")
        if not str(it.get("text", "")).strip():
            errs.append(f"{p}: empty text")
        ev = it.get("evidence") or {}
        if not ev.get("speaker") or not str(ev["speaker"]).startswith("S"):
            errs.append(f"{p}: evidence.speaker must look like 'S017'")
        s, e = ev.get("start"), ev.get("end")
        if not (isinstance(s, (int, float)) and isinstance(e, (int, float)) and lo <= s < e <= hi + 1):
            errs.append(f"{p}: evidence start/end invalid or outside the window")
        if it.get("lang_of_evidence") not in LANGS:
            errs.append(f"{p}: lang_of_evidence must be one of {sorted(LANGS)}")
        unk = set(it.get("unknown_fields", []))
        if it.get("type") == "action":
            for f in ("owner", "deadline"):
                if it.get(f) is None and f not in unk:
                    errs.append(f"{p}: {f} is null but not listed in unknown_fields")
        for f in unk:
            if it.get(f) is not None:
                errs.append(f"{p}: {f} is listed unknown but has a value")
    return a, errs


def main():
    root = load_config()["paths"]["annotations"]
    bad = 0
    for path in sorted(root.glob("*/annotations.json")):
        a, errs = check(path)
        n = len(a["items"])
        print(f"{path.parent.name}: {n} items, {len(errs)} problems"
              + ("" if n else "  (not annotated yet)"))
        for e in errs:
            print("   -", e)
        bad += len(errs)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
