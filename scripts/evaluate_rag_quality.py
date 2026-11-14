from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z]+", text.lower()))


def overlap_ratio(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def evaluate_run(run_dir: Path, eval_rows: List[Dict], kb: Dict[str, str]) -> Dict[str, float]:
    preds = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in eval_rows}

    faithfulness = []
    answer_relevancy = []
    context_precision = []
    for p in preds:
        item = by_id[p["id"]]
        ans = p["answer"]
        qry = item["query"]
        retrieved_ids = p["retrieved_ids"]
        retrieved_text = " ".join(kb[i] for i in retrieved_ids if i in kb)
        faithfulness.append(overlap_ratio(ans, retrieved_text))
        answer_relevancy.append(overlap_ratio(ans, qry))
        useful = set(item["gold_context"])
        if retrieved_ids:
            context_precision.append(len(set(retrieved_ids) & useful) / len(retrieved_ids))
        else:
            context_precision.append(0.0)
    n = max(len(preds), 1)
    return {
        "faithfulness_proxy": sum(faithfulness) / n,
        "answer_relevancy_proxy": sum(answer_relevancy) / n,
        "context_precision_proxy": sum(context_precision) / n,
    }


def main() -> None:
    eval_rows = []
    with (ROOT / "data" / "custom_eval" / "ea_rag_eval.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eval_rows.append(json.loads(line))
    kb = {}
    with (ROOT / "data" / "processed" / "knowledge_base.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                kb[row["id"]] = row["text"]

    baseline = evaluate_run(ROOT / "results" / "baseline", eval_rows, kb)
    earag = evaluate_run(ROOT / "results" / "earag", eval_rows, kb)
    payload = {"baseline": baseline, "earag": earag}
    out = ROOT / "results" / "rag_quality_metrics.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
