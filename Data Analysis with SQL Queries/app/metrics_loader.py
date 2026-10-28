"""
Objective 3 support: loads the real saved metrics_*.json files produced by
MetricsCollector (utils/metrics_collector.py) and turns them into a flat
pandas DataFrame for analysis. Nothing here invents numbers — every row
comes from a file that was actually written by a real benchmark run.
"""
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from app.framework_loader import RESULTS_DIR

# NOTE on "api_calls": MetricsCollector.increment_api_calls() is called once
# per QUESTION in both kickoff_crew() and build_graph() (see the original
# scripts), not once per underlying LLM request. So this field is really a
# "tasks completed" counter that always equals the number of questions in
# the iteration — it is NOT a true LLM API request count. We surface it
# under an honest label and never call it "API calls" in the UI.
TASKS_LABEL = "tasks_completed_per_iteration"


def discover_metrics_files() -> List[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("metrics_*.json"))


def load_all_runs() -> pd.DataFrame:
    """Flattens every saved metrics_*.json into one row per (file, framework, iteration)."""
    rows = []
    for path in discover_metrics_files():
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        for framework, iterations in data.items():
            for entry in iterations:
                if entry.get("iteration") == "average":
                    continue  # summary row, not a real iteration
                api_latencies = entry.get("api_latencies", [])
                rows.append({
                    "source_file": path.name,
                    "framework": framework,
                    "iteration": entry.get("iteration"),
                    "latency_s": entry.get("latency"),
                    TASKS_LABEL: entry.get("api_calls"),
                    "retries": entry.get("retries", 0),
                    "avg_api_latency_s": entry.get("avg_api_latency"),
                    "match_percentage": entry.get("match_percentage"),
                    "n_api_latency_samples": len(api_latencies),
                    "min_api_latency_s": min(api_latencies) if api_latencies else None,
                    "max_api_latency_s": max(api_latencies) if api_latencies else None,
                })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-framework summary stats, computed only from what's actually in df."""
    if df.empty:
        return pd.DataFrame()

    summaries = []
    for framework, group in df.groupby("framework"):
        n = len(group)
        latencies = group["latency_s"].dropna().tolist()
        accuracies = group["match_percentage"].dropna().tolist()
        summaries.append({
            "framework": framework,
            "runs_recorded": n,
            "avg_latency_s": statistics.mean(latencies) if latencies else None,
            "median_latency_s": statistics.median(latencies) if latencies else None,
            "latency_stdev_s": statistics.stdev(latencies) if len(latencies) >= 2 else None,
            "avg_accuracy_pct": statistics.mean(accuracies) if accuracies else None,
            "accuracy_stdev_pct": statistics.stdev(accuracies) if len(accuracies) >= 2 else None,
            "avg_tasks_per_iteration": statistics.mean(group[TASKS_LABEL].dropna().tolist()) if group[TASKS_LABEL].notna().any() else None,
            "total_retries": int(group["retries"].fillna(0).sum()),
        })
    return pd.DataFrame(summaries)


def interpretation_text(summary_df: pd.DataFrame) -> str:
    """Short, dynamically generated interpretation — no hardcoded framework winner."""
    if summary_df.empty:
        return "No saved benchmark runs were found in results/, so no comparison can be generated yet."

    lines = []
    frameworks = summary_df["framework"].tolist()
    lines.append(
        f"Based on {int(summary_df['runs_recorded'].sum())} recorded iteration(s) "
        f"across {', '.join(frameworks)}:"
    )
    for _, row in summary_df.iterrows():
        lat = f"{row['avg_latency_s']:.2f}s" if row["avg_latency_s"] is not None else "unavailable"
        lat_sd = f" (σ={row['latency_stdev_s']:.2f}s)" if row["latency_stdev_s"] is not None else ""
        acc = f"{row['avg_accuracy_pct']:.1f}%" if row["avg_accuracy_pct"] is not None else "unavailable"
        lines.append(f"- **{row['framework']}**: avg iteration latency {lat}{lat_sd}, avg accuracy {acc}, "
                      f"{row['runs_recorded']} run(s) recorded.")

    if len(summary_df) >= 2 and summary_df["avg_latency_s"].notna().all():
        faster = summary_df.loc[summary_df["avg_latency_s"].idxmin(), "framework"]
        lines.append(f"\n{faster} was faster on average in this recorded data, but with only "
                      f"{int(summary_df['runs_recorded'].min())}-{int(summary_df['runs_recorded'].max())} "
                      f"iterations per framework this is not a statistically strong claim — "
                      f"neither framework should be read as universally superior from this sample.")
    return "\n".join(lines)
