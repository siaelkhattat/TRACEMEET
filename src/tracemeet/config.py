"""Load config. Set TRACEMEET_DATA_ROOT (e.g. a Google Drive folder) in Colab
so raw data and outputs persist; annotations/splits always stay in the repo."""
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
REPO_KEYS = ("annotations", "splits")


def load_config(path=ROOT / "configs" / "config.yaml"):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_root = Path(os.environ.get("TRACEMEET_DATA_ROOT", ROOT))
    cfg["paths"] = {
        k: (ROOT if k in REPO_KEYS else data_root) / v
        for k, v in cfg["paths"].items()
    }
    return cfg
