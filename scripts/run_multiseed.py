from __future__ import annotations

import json
import subprocess
from pathlib import Path
from statistics import mean, stdev

import yaml


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "experiment.yaml"


def load_cfg() -> dict:
    return yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))


def save_cfg(payload: dict) -> None:
    CFG_PATH.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def run_once(seed: int) -> dict:
    cfg = load_cfg()
    cfg["experiment"]["seed"] = seed
    save_cfg(cfg)
    subprocess.run(["python", "scripts/run_baseline.py"], cwd=ROOT, check=True)
    subprocess.run(["python", "scripts/run_earag.py"], cwd=ROOT, check=True)
    b = json.loads((ROOT / "results" / "baseline" / "metrics.json").read_text(encoding="utf-8"))
    e = json.loads((ROOT / "results" / "earag" / "metrics.json").read_text(encoding="utf-8"))
    return {"seed": seed, "baseline": b, "earag": e}


def aggregate(rows: list[dict], metric: str, system: str) -> dict:
    values = [r[system][metric] for r in rows]
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": mean(values), "std": stdev(values)}


def main() -> None:
    cfg = load_cfg()
    seeds = cfg["experiment"].get("seeds", [cfg["experiment"]["seed"]])
    original_seed = cfg["experiment"]["seed"]
    rows = [run_once(s) for s in seeds]

    summary = {
        "seeds": seeds,
        "baseline": {m: aggregate(rows, m, "baseline") for m in ["hit@1", "hit@3", "hit@5"]},
        "earag": {m: aggregate(rows, m, "earag") for m in ["hit@1", "hit@3", "hit@5"]},
        "raw": rows,
    }
    (ROOT / "results" / "multiseed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cfg["experiment"]["seed"] = original_seed
    save_cfg(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
