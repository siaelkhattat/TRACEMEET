# Annotation guidelines (decisions and actions)

Annotate from the gold transcript (`transcript.md`), window 0-15 min.
**Annotate before looking at any system output.**

## Item types
- `decision`: the group explicitly commits to a choice ("ok, on prend la comédie").
  A suggestion is not a decision until someone confirms and nobody objects.
- `action`: someone is assigned, or volunteers, to do something after the meeting.
- `rejected_proposal`: a proposal explicitly refused or withdrawn.
- Late reversal: annotate the FINAL state as the decision, and the abandoned option
  as `rejected_proposal`.
- Zero items is a valid answer for a meeting. Do not invent items.

## Fields
- `text`: short French description of the item, in your own words.
- `owner`: speaker id as written in the transcript ("S017"); `"external:<name>"` if a
  non-participant; `null` if not said. Actions only (null is normal for decisions).
- `deadline`: copied as spoken ("avant vendredi"); `null` if not said.
- `unknown_fields`: names of fields left null that matter for this item type
  (actions: "owner" and/or "deadline"). Do not guess missing information.
- `evidence`: `{"speaker": "S017", "start": 312.4, "end": 318.9}`: the utterance where
  the commitment is made or confirmed (seconds from window start, as in the transcript).
- `lang_of_evidence`: `fr` | `darija` | `mixed`.

## Example item
{"type": "action", "text": "Préparer la feuille de budget", "owner": "S018",
 "deadline": "avant vendredi", "unknown_fields": [],
 "evidence": {"speaker": "S018", "start": 312.4, "end": 318.9}, "lang_of_evidence": "fr"}

## Quality
Double-annotate at least one meeting if a second person is available, and report
agreement. Otherwise state the single-annotator limitation.
