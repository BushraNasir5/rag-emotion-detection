from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fmt(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.3f} +- {std_value:.3f}"


def main() -> None:
    summary_path = ROOT / "results" / "multiseed_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("Run scripts/run_multiseed.py first.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    table_path = ROOT / "paper" / "tables" / "results_main_camera_ready.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)

    with table_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["system", "hit@1", "hit@3", "hit@5"])
        writer.writerow(
            [
                "baseline_rag",
                fmt(summary["baseline"]["hit@1"]["mean"], summary["baseline"]["hit@1"]["std"]),
                fmt(summary["baseline"]["hit@3"]["mean"], summary["baseline"]["hit@3"]["std"]),
                fmt(summary["baseline"]["hit@5"]["mean"], summary["baseline"]["hit@5"]["std"]),
            ]
        )
        writer.writerow(
            [
                "ea_rag",
                fmt(summary["earag"]["hit@1"]["mean"], summary["earag"]["hit@1"]["std"]),
                fmt(summary["earag"]["hit@3"]["mean"], summary["earag"]["hit@3"]["std"]),
                fmt(summary["earag"]["hit@5"]["mean"], summary["earag"]["hit@5"]["std"]),
            ]
        )

    md_path = ROOT / "paper" / "tables" / "results_main_camera_ready.md"
    md = [
        "| System | Hit@1 | Hit@3 | Hit@5 |",
        "|---|---:|---:|---:|",
        f"| Baseline RAG | {fmt(summary['baseline']['hit@1']['mean'], summary['baseline']['hit@1']['std'])} | {fmt(summary['baseline']['hit@3']['mean'], summary['baseline']['hit@3']['std'])} | {fmt(summary['baseline']['hit@5']['mean'], summary['baseline']['hit@5']['std'])} |",
        f"| EA-RAG | {fmt(summary['earag']['hit@1']['mean'], summary['earag']['hit@1']['std'])} | {fmt(summary['earag']['hit@3']['mean'], summary['earag']['hit@3']['std'])} | {fmt(summary['earag']['hit@5']['mean'], summary['earag']['hit@5']['std'])} |",
        "",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {table_path} and {md_path}")


if __name__ == "__main__":
    main()
