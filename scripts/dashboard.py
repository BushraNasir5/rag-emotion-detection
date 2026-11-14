from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    st.set_page_config(page_title="EA-RAG Dashboard", layout="wide")
    st.title("EA-RAG Results Dashboard")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("### Quick links (files)")
        st.code("results/baseline/metrics.json\nresults/earag/metrics.json\nresults/ablation_metrics.json\nresults/multiseed_summary.json\npaper/tables/results_main_camera_ready.md", language="text")

    with colB:
        st.markdown("### Current status")
        ds = read_json(ROOT / "data" / "processed" / "dataset_summary.json")
        if ds:
            st.json(ds)
        else:
            st.info("No `data/processed/dataset_summary.json` yet. Run `python scripts/prepare_real_datasets.py`.")

    st.markdown("### Baseline vs EA-RAG (single run)")
    baseline = read_json(ROOT / "results" / "baseline" / "metrics.json") or {}
    earag = read_json(ROOT / "results" / "earag" / "metrics.json") or {}
    df = pd.DataFrame(
        [
            {"system": "baseline", **baseline},
            {"system": "earag", **earag},
        ]
    ).fillna(0.0)
    st.dataframe(df, use_container_width=True)

    if {"hit@1", "hit@3", "hit@5"}.issubset(df.columns):
        chart_df = df.melt(id_vars=["system"], value_vars=["hit@1", "hit@3", "hit@5"], var_name="metric", value_name="value")
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(x="metric:N", y="value:Q", color="system:N", column="system:N")
            .properties(height=250)
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("### Ablations")
    ablations = read_json(ROOT / "results" / "ablation_metrics.json")
    if ablations:
        ab_df = pd.DataFrame(ablations)
        st.dataframe(ab_df, use_container_width=True)
        if {"hit@1", "hit@3", "hit@5"}.issubset(ab_df.columns):
            ab_melt = ab_df.melt(id_vars=["name"], value_vars=["hit@1", "hit@3", "hit@5"], var_name="metric", value_name="value")
            ab_chart = alt.Chart(ab_melt).mark_line(point=True).encode(x="metric:N", y="value:Q", color="name:N").properties(height=300)
            st.altair_chart(ab_chart, use_container_width=True)
    else:
        st.info("No ablations found. Run `python scripts/run_ablation.py`.")

    st.markdown("### Multi-seed summary (camera-ready)")
    ms = read_json(ROOT / "results" / "multiseed_summary.json")
    if ms:
        rows = []
        for system in ["baseline", "earag"]:
            for metric in ["hit@1", "hit@3", "hit@5"]:
                rows.append(
                    {
                        "system": system,
                        "metric": metric,
                        "mean": ms[system][metric]["mean"],
                        "std": ms[system][metric]["std"],
                    }
                )
        ms_df = pd.DataFrame(rows)
        st.dataframe(ms_df, use_container_width=True)
        ms_chart = alt.Chart(ms_df).mark_bar().encode(x="metric:N", y="mean:Q", color="system:N").properties(height=250)
        st.altair_chart(ms_chart, use_container_width=True)
    else:
        st.info("No `results/multiseed_summary.json`. Run `python scripts/run_multiseed.py`.")

    st.markdown("### Paper table preview")
    table_md = ROOT / "paper" / "tables" / "results_main_camera_ready.md"
    if table_md.exists():
        st.markdown(table_md.read_text(encoding="utf-8"))
    else:
        st.info("No camera-ready markdown table yet. Run `python scripts/generate_camera_ready_tables.py`.")


if __name__ == "__main__":
    main()

