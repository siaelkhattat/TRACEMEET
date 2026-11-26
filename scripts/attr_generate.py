"""Qwen3 (4-bit) writes attributed claims, one 150-second window at a time."""
import argparse
import json
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from tracemeet.attribution import CONDS, format_transcript, load_rows, parse_claims
from tracemeet.config import load_config

SYSTEM = "Tu es un assistant qui rédige des comptes rendus de réunion fidèles et précis."
USER = """Voici un extrait de transcription d'une réunion en français. Chaque ligne a le format : [temps en secondes] Sxxx: texte.

{transcript}

Rédige entre 2 et 3 affirmations courtes et factuelles sur ce que les participants ont dit, proposé ou décidé dans cet extrait. Chaque affirmation doit :
- nommer le locuteur concerné par son identifiant exact (par exemple S017), tel qu'il apparaît dans la transcription ;
- indiquer le temps (en secondes) de la ligne qui la soutient ;
- être rédigée en français.
Réponds uniquement par un tableau JSON, sans autre texte : [{{"speaker": "S017", "time": 123, "claim": "..."}}]"""
WINDOW_S, MAX_IN, MAX_NEW = 150.0, 4500, 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    name = cfg["llm"]["model"]
    sel = json.loads((cfg["paths"]["splits"] / "selected_meetings.json").read_text())
    meetings = [m for s in ("dev", "test") for m in sel[s]]

    tok = AutoTokenizer.from_pretrained(name)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb, device_map="auto")

    def build(tr):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(transcript=tr)}]
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    for mid in meetings:
        for cond in CONDS:
            dest = cfg["paths"]["extraction_out"] / "attribution" / cond / f"{mid}.json"
            if dest.exists() and not args.force:
                print("skip", mid, cond)
                continue
            try:
                rows = load_rows(cfg, mid, cond)
            except FileNotFoundError:
                print(f"!! missing ASR output for {mid} / {cond}, skipping")
                continue
            groups = defaultdict(list)
            for r in rows:
                groups[int(r["start"] // WINDOW_S)].append(r)

            t0, claims, raws, max_tok, n_trunc = time.time(), [], [], 0, 0
            for w in sorted(groups):
                tr = format_transcript(groups[w])
                if not tr.strip():
                    continue
                prompt = build(tr)
                n_in = len(tok(prompt).input_ids)
                if n_in > MAX_IN:
                    tr = tr[: int(len(tr) * MAX_IN / n_in * 0.95)]
                    prompt, n_trunc = build(tr), n_trunc + 1
                    n_in = len(tok(prompt).input_ids)
                max_tok = max(max_tok, n_in)
                inputs = tok(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
                text = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                raws.append(text)
                claims += parse_claims(text)
                del out, inputs
                torch.cuda.empty_cache()

            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps({"meeting": mid, "cond": cond, "max_tokens_in": max_tok,
                                        "truncated_windows": n_trunc,
                                        "runtime_s": round(time.time() - t0, 1),
                                        "raw_outputs": raws, "claims": claims},
                                       ensure_ascii=False, indent=1))
            print(f"{mid} {cond}: {len(claims)} claims, max {max_tok} tokens, "
                  f"{time.time() - t0:.0f}s" + (f" ({n_trunc} truncated)" if n_trunc else ""))
            if not claims:
                print("   !! no claims parsed, check raw_outputs in", dest)


if __name__ == "__main__":
    main()
