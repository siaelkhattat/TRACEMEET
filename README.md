# TraceMeet

Reliable French–Darija meeting summarization with speaker attribution,
evidence-grounded decisions and uncertainty awareness (TALN module project).

## Layout
- `src/tracemeet/` pipeline code and evaluation
- `configs/config.yaml` models and paths
- `data/annotations/` gold annotations (tracked); `data/raw`, `outputs/` are not tracked
- `scenarios/` briefs for the simulated French–Darija meetings
- `notebooks/` Colab runners

## Workflow
Code locally in PyCharm, run GPU steps in Colab (clone repo, set `TRACEMEET_DATA_ROOT` to a Drive folder).
