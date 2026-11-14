from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: str | Path) -> List[Dict]:
    records: List[Dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_json(path: str | Path, payload: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
