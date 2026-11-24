"""Text normalization for WER/CER. SUMM-RE gold follows SPPAS conventions."""
import re

from num2words import num2words

DROP_TOKENS = {"*", "+", "@"}  # noise, short pause, laughter
FILLERS = {"euh", "heu", "hum", "mh", "mhm", "hm", "hmm"}  # pure hesitations


def strip_gold_markup(t):
    t = re.sub(r"\{[^}]*\}", " ", t)                    # {comments}
    t = re.sub(r"\[([^\],]*),[^\]]*\]", r"\1", t)       # [orth,pron] -> orth
    t = re.sub(r"\[[^\]]*\]", " ", t)                   # [comment]
    t = re.sub(r"<([^<>,]*),[^<>]*>", r"\1", t)         # <a,b> -> a
    t = t.replace("$", " ")                             # proper-name markers
    t = re.sub(r"=\w*=", " ", t)                        # liaisons
    t = re.sub(r"[()]", "", t)                          # elisions: keep letters
    toks = [k for k in t.split()
            if k not in DROP_TOKENS and not (len(k) > 1 and k.endswith("-"))]
    return " ".join(toks)                               # truncated words dropped


def _common(t, drop_fillers=False):
    t = t.lower().replace("’", "'")
    t = re.sub(r"\d+", lambda m: " " + num2words(int(m.group()), lang="fr") + " ", t)
    t = t.replace("-", " ")
    t = re.sub(r"[^\w\s']", " ", t)
    t = t.replace("'", "' ")
    t = re.sub(r"\s+", " ", t).strip()
    if drop_fillers:
        t = " ".join(w for w in t.split() if w not in FILLERS)
    return t


def normalize_gold(t, drop_fillers=False):
    return _common(strip_gold_markup(t), drop_fillers)


def normalize_hyp(t, drop_fillers=False):
    return _common(t, drop_fillers)
