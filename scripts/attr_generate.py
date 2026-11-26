"""Qwen3-8B (4-bit) writes attributed claims from gold / masked-ASR / raw-ASR transcripts."""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from tracemeet.attribution import CONDS, format_transcript, load_rows, parse_claims
from tracemeet.config import load_config

SYSTEM = "Tu es un assistant qui rédige des comptes rendus de réunion fidèles et précis."
USER = """Voici la transcription d'une réunion en français. Chaque ligne a le format : [temps en secondes] Sxxx: texte.

{transcript}

Rédige entre 12 et 16 affirmations courtes et factuelles sur ce que les participants ont dit, proposé ou décidé. Chaque affirmation doit :
- nommer le locuteur concerné par son identifiant exact (par exemple S017), tel qu'il apparaît dans la transcription ;
- indiquer le temps (en secondes) de la ligne qui la soutient ;
- être rédigée en français.
Réponds uniquement par un tableau JSON, sans autre texte : [{{"speaker": "S017", "time": 123, "claim": "..."}}]"""
MAX_IN = 20000


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

    for mid in meetings:
        for cond in CONDS:
            dest = cfg["paths"]["extraction_out"] / "attribution" / cond / f"{mid}.json"
            if dest.exists() and not args.force:
                print("skip", mid, cond)
                continue
            try:
                transcript = format_transcript(load_rows(cfg, mid, cond))
            except FileNotFoundError:
                print(f"!! missing ASR output for {mid} / {cond}, skipping")
                continue

            def build(tr):
                msgs = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": USER.format(transcript=tr)}]
                return tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)

            prompt = build(transcript)
            n_in = len(tok(prompt).input_ids)
            truncated = n_in > MAX_IN
            if truncated:
                transcript = transcript[: int(len(transcript) * MAX_IN / n_in * 0.95)]
                prompt = build(transcript)
                n_in = len(tok(prompt).input_ids)
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=1500, do_sample=False)
            text = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            claims = parse_claims(text)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps({"meeting": mid, "cond": cond, "n_tokens_in": n_in,
                                        "truncated": truncated, "runtime_s": round(time.time() - t0, 1),
                                        "raw_output": text, "claims": claims},
                                       ensure_ascii=False, indent=1))
            print(f"{mid} {cond}: {len(claims)} claims, {n_in} tokens in, "
                  f"{time.time() - t0:.0f}s" + (" (TRUNCATED)" if truncated else ""))
            if not claims:
                print("   !! no claims parsed, check raw_output in", dest)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
