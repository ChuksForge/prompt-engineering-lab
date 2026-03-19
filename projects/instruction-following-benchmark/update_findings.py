"""
update_findings.py
==================
Instruction Following Benchmark — Auto-populate findings
Project: P3 · prompt-engineering-lab

Run AFTER run_experiment.py to auto-fill:
  - README.md leaderboard table + failure analysis table
  - Notebook Cell findings (printed — paste in)

Usage:
    python update_findings.py
"""

import json
from pathlib import Path
from collections import Counter

import pandas as pd

RESULTS_DIR = Path("results")
README_PATH = Path("README.md")

CATEGORY_LABELS = {
    "multi_step":   "Multi-Step",
    "tone_persona": "Tone & Persona",
    "negation":     "Negation Handling",
}


def load() -> tuple:
    df  = pd.read_csv(RESULTS_DIR / "results.csv")
    lb  = pd.read_csv(RESULTS_DIR / "leaderboard.csv")
    fr  = pd.read_csv(RESULTS_DIR / "failure_report.csv")
    df  = df[df["error"].isna() | (df["error"] == "")].copy()
    return df, lb, fr


def build_leaderboard_table(lb: pd.DataFrame) -> str:
    overall = lb[lb["category"] == "OVERALL"].sort_values("avg_pass_rate", ascending=False)
    cats = lb[lb["category"] != "OVERALL"]

    # Category pass rates per model
    cat_pivot = cats.pivot_table(
        index="model", columns="category", values="avg_pass_rate"
    ).fillna(0).round(3)

    rows = [
        "| Rank | Model | Overall | Multi-Step | Tone & Persona | Negation | Full Compliance |",
        "|------|-------|---------|------------|----------------|----------|-----------------|",
    ]

    for rank, (_, row) in enumerate(overall.iterrows(), 1):
        model = row["model"]
        overall_rate = row["avg_pass_rate"]
        full_rate    = row["compliance_rate"]
        ms   = cat_pivot.loc[model, "multi_step"]   if model in cat_pivot.index and "multi_step"   in cat_pivot.columns else 0
        tp   = cat_pivot.loc[model, "tone_persona"] if model in cat_pivot.index and "tone_persona" in cat_pivot.columns else 0
        ng   = cat_pivot.loc[model, "negation"]     if model in cat_pivot.index and "negation"     in cat_pivot.columns else 0

        rows.append(
            f"| {rank} | {model} | {overall_rate:.1%} "
            f"| {ms:.1%} | {tp:.1%} | {ng:.1%} | {full_rate:.1%} |"
        )

    return "\n".join(rows)


def build_failure_table(fr: pd.DataFrame) -> str:
    if fr.empty:
        return "_No failure data available._"

    rows = [
        "| Model | Top Failure Mode | 2nd Mode | 3rd Mode |",
        "|-------|-----------------|----------|----------|",
    ]
    for model in fr["model"].unique():
        mfr = fr[fr["model"] == model].sort_values("count", ascending=False)
        modes = mfr["failure_mode"].tolist()
        pcts  = mfr["pct_of_failures"].tolist()

        def fmt(i):
            if i < len(modes):
                return f"{modes[i]} ({pcts[i]:.0f}%)"
            return "—"

        rows.append(f"| {model} | {fmt(0)} | {fmt(1)} | {fmt(2)} |")

    return "\n".join(rows)


def update_readme(lb_table: str, fail_table: str):
    content = README_PATH.read_text(encoding="utf-8")

    # Replace leaderboard table
    lb_start = "| Rank | Model | Overall |"
    lb_end   = "\n\n*Run"
    si = content.find(lb_start)
    ei = content.find(lb_end, si)
    if si != -1 and ei != -1:
        content = content[:si] + lb_table + content[ei:]
        print("  README leaderboard table updated.")

    # Replace failure table
    ft_start = "| Model | Top Failure Mode |"
    ft_end_marker = "\n\n---"
    si2 = content.find(ft_start)
    ei2 = content.find(ft_end_marker, si2)
    if si2 != -1 and ei2 != -1:
        content = content[:si2] + fail_table + content[ei2:]
        print("  README failure table updated.")

    README_PATH.write_text(content, encoding="utf-8")


def print_notebook_findings(df: pd.DataFrame, lb: pd.DataFrame, fr: pd.DataFrame):
    overall = lb[lb["category"] == "OVERALL"].sort_values("avg_pass_rate", ascending=False)
    best_model    = overall.iloc[0]["model"]
    best_rate     = overall.iloc[0]["avg_pass_rate"]
    worst_model   = overall.iloc[-1]["model"]
    worst_rate    = overall.iloc[-1]["avg_pass_rate"]
    best_full     = overall.iloc[0]["compliance_rate"]

    cats = lb[lb["category"] != "OVERALL"]
    cat_avg = cats.groupby("category")["avg_pass_rate"].mean()
    hardest_cat   = cat_avg.idxmin()
    hardest_rate  = cat_avg.min()
    easiest_cat   = cat_avg.idxmax()
    easiest_rate  = cat_avg.max()

    diff_avg = df.groupby("difficulty")["pass_rate"].mean()
    easy_r  = diff_avg.get("easy",  0)
    hard_r  = diff_avg.get("hard",  0)
    drop    = easy_r - hard_r

    top_fail = fr.groupby("failure_mode")["count"].sum().sort_values(ascending=False)
    top_fail_mode = top_fail.index[0] if not top_fail.empty else "N/A"
    top_fail_pct  = top_fail.iloc[0] / top_fail.sum() * 100 if not top_fail.empty else 0

    print("\n" + "="*62)
    print("COPY THIS INTO NOTEBOOK KEY FINDINGS CELL:")
    print("="*62)
    print(f"""
## Key Findings

1. **Best model overall:** `{best_model}` — {best_rate:.1%} avg pass rate, {best_full:.1%} full compliance
2. **Worst model overall:** `{worst_model}` — {worst_rate:.1%} avg pass rate
3. **Hardest category:** `{CATEGORY_LABELS.get(hardest_cat, hardest_cat)}` — {hardest_rate:.1%} avg pass rate
4. **Easiest category:** `{CATEGORY_LABELS.get(easiest_cat, easiest_cat)}` — {easiest_rate:.1%} avg pass rate
5. **Difficulty gap:** Easy={easy_r:.1%} → Hard={hard_r:.1%} — {drop:.1%} drop in pass rate
6. **Most common failure mode:** `{top_fail_mode}` — {top_fail_pct:.0f}% of all failures
7. **Key insight:** [Fill in after reviewing failure_report.csv and task heatmap]

---
*See `results/failure_report.csv` for per-model failure breakdown.*
*See `results/chart_task_heatmap.png` for task-level pass rates.*
""")


def main():
    print("Loading results...")
    df, lb, fr = load()
    print(f"  {len(df)} rows | {df['model'].nunique()} models | {df['task_id'].nunique()} tasks\n")

    print("Building README tables...")
    lb_table   = build_leaderboard_table(lb)
    fail_table = build_failure_table(fr)
    update_readme(lb_table, fail_table)

    print_notebook_findings(df, lb, fr)
    print("\nDone. Commit your updated README.md.")


if __name__ == "__main__":
    main()
