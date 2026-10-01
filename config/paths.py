"""Resolve the shared path configuration relative to the repository root."""
from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[1]
def input_path(key):
    with (ROOT / "config/paths.json").open(encoding="utf-8") as stream:
        value = json.load(stream)[key]
    path = Path(value)
    if path.is_absolute():
        raise ValueError("Paths must be relative to the repository root: " + key)
    return ROOT / path
