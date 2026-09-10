"""Structure-name matching shared by the Python evaluation tools
(source/gui_metrics.py, source/batch_evaluate.py) and the GUI's Metrics table
-- gui/src/structure_metrics.cpp implements normalize/match_key identically.

The same structure is spelled differently by different sources: "OpticNrv_L"
(HaN-Seg), "Optic Nerve-Lt" (the phantom's RTSTRUCT), "optic_nerve_l" (the
model's dataset.json). Two names refer to the same structure when their match
keys are equal: the normalized name (lowercase alphanumerics only), mapped
through configs/structure_aliases.json when that has an entry for it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ALIASES_PATH = Path("configs/structure_aliases.json")


def normalize(name: str) -> str:
    return re.sub(r"[^0-9a-z]", "", name.lower())


def load_aliases(path: Path = ALIASES_PATH) -> dict[str, str]:
    """{normalized name: normalized target}; empty when the file doesn't exist."""
    if not path.exists():
        return {}
    aliases = json.loads(path.read_text()).get("aliases", {})
    return {normalize(src): normalize(dst) for src, dst in aliases.items() if normalize(src) and normalize(dst)}


def match_key(name: str, aliases: dict[str, str] | None = None) -> str:
    key = normalize(name)
    return (load_aliases() if aliases is None else aliases).get(key, key)
