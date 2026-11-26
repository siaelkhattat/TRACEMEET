"""Shared pipeline steps: LLM correction, then structured minutes (summary, decisions, actions)."""
import copy
import json
import re
from collections import defaultdict

from tracemeet.attribution import format_transcript
from tracemeet.llm import chat

WINDOW_S = 150.0
MEETINGS = ["006b_EADH", "025b_EBDH", "020b_EBDZ"]

SYS_FIX = "Tu es un correcteur de transcriptions automatiques de réunions en français."
USER_FIX = """Voici des lignes d'une transcription automatique (Whisper). Format : [numéro] Slocuteur: texte.

{lines}

Corrige uniquement les erreurs de reconnaissance évidentes : mots mal reconnus ou homophones, accords, noms propres, élisions. N'ajoute rien, ne supprime rien, ne reformule pas, garde les hésitations. Si tu n'es pas sûr, laisse le texte tel quel.
Réponds avec exactement le même nombre de lignes, au format : [numéro] texte corrigé (sans le locuteur)."""

SYS_SUM = "Tu es un assistant qui rédige des comptes rendus de réunion fidèles et précis."
USER_SUM = """Voici un extrait de transcription d'une réunion en français. Format : [temps en secondes] Slocuteur: texte.

{transcript}

Rédige un compte rendu de cet extrait en JSON, sans autre texte :
{{"resume": "2 à 3 phrases", "decisions": ["décision prise, en une phrase"], "actions": [{{"qui": "S017 ou null", "quoi": "tâche à faire", "echeance": "délai cité ou null"}}]}}
Ne note que ce qui est réellement dit. Listes vides s'il n'y a ni décision ni action."""

_LINE = re.compile(r"^\[(\d+)\]\s*(.*)$")


def make_windows(rows, window_s=WINDOW_S, limit_s=float("inf")):
    groups = defaultdict(list)
    for r in rows:
        if r["start"] < limit_s:
            groups[int(r["start"] // window_s)].append(r)
    return [groups[k] for k in sorted(groups)]


def correct_window(tok, model, rows, min_ratio=0.6, max_ratio=1.5):
    """LLM correction, line by line. A correction that changes the length too much is
    rejected (guard against over-correction / hallucinated insertions)."""
    lines = "\n".join(f"[{i}] S{r['speaker']}: {r['text']}" for i, r in enumerate(rows))
    raw = chat(tok, model, SYS_FIX, USER_FIX.format(lines=lines), max_new=2000)
    fixed = {}
    for ln in raw.splitlines():
        m = _LINE.match(ln.strip())
        if m:
            fixed[int(m.group(1))] = re.sub(r"^S\d+\s*:\s*", "", m.group(2)).strip()
    out, rejected = [], 0
    for i, r in enumerate(rows):
        new = copy.deepcopy(r)
        new["text_orig"] = r["text"]
        t = fixed.get(i)
        if t:
            n0, n1 = len(r["text"].split()), len(t.split())
            ok = n0 == 0 or (n0 <= 3 and n1 <= n0 + 2) or (min_ratio <= n1 / n0 <= max_ratio)
            if ok:
                new["text"] = t
            else:
                rejected += 1
        out.append(new)
    return out, rejected, raw


def summarize_window(tok, model, rows):
    raw = chat(tok, model, SYS_SUM, USER_SUM.format(transcript=format_transcript(rows)), max_new=600)
    m = re.search(r"\{.*\}", raw, flags=re.S)
    data = {}
    if m:
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            data = {}
    dec = data.get("decisions", [])
    dec = [str(d).strip() for d in dec if str(d).strip()] if isinstance(dec, list) else []
    acts = []
    if isinstance(data.get("actions"), list):
        for a in data["actions"]:
            if isinstance(a, dict) and str(a.get("quoi", "")).strip():
                acts.append({"qui": a.get("qui"), "quoi": str(a["quoi"]).strip(),
                             "echeance": a.get("echeance")})
    return {"resume": str(data.get("resume", "")).strip(), "decisions": dec, "actions": acts,
            "parsed": bool(data), "raw": raw}
