from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "experiment.yaml"


def write_cfg(remove_query_rewrite: bool, remove_emotion_ranking: bool, remove_tone_conditioning: bool) -> None:
    import yaml

    with CFG.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    payload["experiment"]["ablations"]["remove_query_rewrite"] = remove_query_rewrite
    payload["experiment"]["ablations"]["remove_emotion_ranking"] = remove_emotion_ranking
    payload["experiment"]["ablations"]["remove_tone_conditioning"] = remove_tone_conditioning
    with CFG.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def run_case(name: str, rq: bool, rr: bool, rt: bool) -> dict:
    write_cfg(rq, rr, rt)
    subprocess.run(["python", "scripts/run_earag.py"], cwd=ROOT, check=True)
    with (ROOT / "results" / "earag" / "metrics.json").open("r", encoding="utf-8") as f:
        m = json.load(f)
    return {"name": name, **m}


def main() -> None:
    cases = [
        ("full", False, False, False),
        ("no_query_rewrite", True, False, False),
        ("no_emotion_ranking", False, True, False),
        ("no_tone_conditioning", False, False, True),
    ]
    rows = [run_case(*c) for c in cases]
    write_cfg(False, False, False)
    out = ROOT / "results" / "ablation_metrics.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
