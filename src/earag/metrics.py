from __future__ import annotations

from typing import Dict, List


def hit_at_k(retrieved_doc_ids: List[str], gold_doc_ids: List[str], k: int) -> float:
    top = set(retrieved_doc_ids[:k])
    gold = set(gold_doc_ids)
    return 1.0 if top.intersection(gold) else 0.0


def simple_retrieval_metrics(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {"hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0}
    h1 = sum(hit_at_k(r["retrieved_ids"], r["gold_ids"], 1) for r in rows) / len(rows)
    h3 = sum(hit_at_k(r["retrieved_ids"], r["gold_ids"], 3) for r in rows) / len(rows)
    h5 = sum(hit_at_k(r["retrieved_ids"], r["gold_ids"], 5) for r in rows) / len(rows)
    return {"hit@1": h1, "hit@3": h3, "hit@5": h5}
