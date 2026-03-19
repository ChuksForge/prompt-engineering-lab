"""
visualize.py
============
Instruction Following Benchmark — Chart Generator
Project: P3 · prompt-engineering-lab

Charts:
  1. Overall pass rate leaderboard (by model)
  2. Pass rate by category × model (grouped bar)
  3. Pass rate by difficulty × model (grouped bar)
  4. Failure mode breakdown (stacked bar per model)
  5. Task-level heatmap (model × task_id, pass rate)
  6. Full compliance rate (% tasks fully passed, no partial)
  7. Master 4-panel hero chart → results/charts.png
"""

import json
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

RESULTS_DIR = Path("results")

PALETTE = {
    "bg":      "#0f1117", "surface": "#161820", "border": "#1e2130",
    "text":    "#f0f2f8", "muted":   "#5a6080", "accent": "#e8ff47",
    "blue":    "#47c8ff", "purple":  "#b847ff", "orange": "#ff8c47",
    "green":   "#47ffb2", "red":     "#ff4776", "pink":   "#ff47c8",
}

MODEL_COLORS = {
    "GPT-4o-mini":       PALETTE["blue"],
    "GPT-4o":            PALETTE["accent"],
    "Claude Haiku":      PALETTE["orange"],
    "Claude Sonnet 4.6": PALETTE["purple"],
    "Mistral 7B":        PALETTE["green"],
    "Llama 3 8B":        PALETTE["red"],
}

FAILURE_COLORS = {
    "MISSED_STEP":      PALETTE["orange"],
    "VIOLATED_NEGATION":PALETTE["red"],
    "WRONG_FORMAT":     PALETTE["blue"],
    "WRONG_TONE":       PALETTE["purple"],
    "LENGTH_VIOLATION": PALETTE["pink"],
}

CATEGORY_LABELS = {
    "multi_step":   "Multi-Step",
    "tone_persona": "Tone & Persona",
    "negation":     "Negation Handling",
}

def setup_style():
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],   "axes.facecolor":  PALETTE["surface"],
        "axes.edgecolor":   PALETTE["border"],"axes.labelcolor": PALETTE["text"],
        "axes.titlecolor":  PALETTE["text"],  "xtick.color":     PALETTE["muted"],
        "ytick.color":      PALETTE["muted"], "text.color":      PALETTE["text"],
        "grid.color":       PALETTE["border"],"grid.linewidth":  0.5,
        "font.family":      "monospace",      "font.size":       9,
        "axes.titlesize":   10,               "axes.titleweight": "bold",
        "axes.titlepad":    10,               "figure.dpi":      150,
    })

def load_data():
    df = pd.read_csv(RESULTS_DIR / "results.csv")
    df = df[df["error"].isna() | (df["error"] == "")].copy()
    lb = pd.read_csv(RESULTS_DIR / "leaderboard.csv")
    fr = pd.read_csv(RESULTS_DIR / "failure_report.csv")
    return df, lb, fr


# ── Chart 1: Overall Leaderboard ────────────────────────────

def chart_leaderboard(lb: pd.DataFrame, out: Path):
    overall = lb[lb["category"] == "OVERALL"].sort_values("avg_pass_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [MODEL_COLORS.get(m, PALETTE["blue"]) for m in overall["model"]]
    bars = ax.barh(overall["model"], overall["avg_pass_rate"],
                   color=colors, height=0.55, alpha=0.9)

    for bar, val, comp in zip(bars, overall["avg_pass_rate"], overall["compliance_rate"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.1%}  (full: {comp:.0%})",
                va="center", ha="left", color=PALETTE["text"], fontsize=8)

    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Average Constraint Pass Rate")
    ax.set_title("OVERALL LEADERBOARD — Avg Pass Rate (full compliance % in brackets)", loc="left")
    ax.axvline(overall["avg_pass_rate"].mean(), color=PALETTE["muted"],
               linestyle="--", linewidth=0.8, alpha=0.6, label="Mean")
    ax.legend(framealpha=0, fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    path = out / "chart_leaderboard.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ {path}")


# ── Chart 2: Pass Rate by Category ──────────────────────────

def chart_by_category(lb: pd.DataFrame, out: Path):
    cats = lb[lb["category"] != "OVERALL"]
    pivot = cats.pivot_table(index="category", columns="model",
                             values="avg_pass_rate", aggfunc="mean").fillna(0)
    pivot.index = [CATEGORY_LABELS.get(c, c) for c in pivot.index]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(pivot.index))
    n = len(pivot.columns)
    width = 0.75 / n

    for i, model in enumerate(pivot.columns):
        offset = (i - n/2) * width + width/2
        color = MODEL_COLORS.get(model, PALETTE["blue"])
        bars = ax.bar(x + offset, pivot[model], width=width*0.9,
                      color=color, alpha=0.85, label=model)
        for bar in bars:
            h = bar.get_height()
            if h > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.0%}", ha="center", va="bottom", fontsize=7,
                        color=PALETTE["muted"])

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Avg Constraint Pass Rate")
    ax.set_title("PASS RATE BY CATEGORY — All Models", loc="left")
    ax.legend(framealpha=0, fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    path = out / "chart_by_category.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ {path}")


# ── Chart 3: Pass Rate by Difficulty ────────────────────────

def chart_by_difficulty(df: pd.DataFrame, out: Path):
    pivot = df.groupby(["difficulty","model"])["pass_rate"].mean().unstack(fill_value=0)
    diff_order = ["easy","medium","hard"]
    pivot = pivot.reindex([d for d in diff_order if d in pivot.index])

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(pivot.index))
    n = len(pivot.columns)
    width = 0.75 / n

    for i, model in enumerate(pivot.columns):
        offset = (i - n/2) * width + width/2
        color = MODEL_COLORS.get(model, PALETTE["blue"])
        ax.bar(x + offset, pivot[model], width=width*0.9,
               color=color, alpha=0.85, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in pivot.index], fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Avg Constraint Pass Rate")
    ax.set_title("PASS RATE BY DIFFICULTY — Performance degradation as complexity increases", loc="left")
    ax.legend(framealpha=0, fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    path = out / "chart_by_difficulty.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ {path}")


# ── Chart 4: Failure Mode Breakdown ─────────────────────────

def chart_failure_modes(fr: pd.DataFrame, out: Path):
    if fr.empty:
        return

    models = fr["model"].unique()
    modes  = list(FAILURE_COLORS.keys())

    # Build matrix: model × failure_mode → count
    matrix = pd.DataFrame(index=models, columns=modes, data=0.0)
    for _, row in fr.iterrows():
        if row["failure_mode"] in modes:
            matrix.loc[row["model"], row["failure_mode"]] = row["count"]

    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(models))
    x = np.arange(len(models))

    for mode in modes:
        color = FAILURE_COLORS.get(mode, PALETTE["muted"])
        vals = matrix[mode].values.astype(float)
        ax.bar(x, vals, bottom=bottom, color=color, alpha=0.85, label=mode, width=0.55)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Failure Count")
    ax.set_title("FAILURE MODE BREAKDOWN — What causes failures per model", loc="left")
    ax.legend(framealpha=0, fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    path = out / "chart_failure_modes.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ {path}")


# ── Chart 5: Task-Level Heatmap ─────────────────────────────

def chart_task_heatmap(df: pd.DataFrame, out: Path):
    pivot = df.pivot_table(index="model", columns="task_id",
                           values="pass_rate", aggfunc="mean").fillna(0)

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns)*0.8), len(pivot.index)*0.7 + 1.5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "black" if val > 0.6 else PALETTE["text"]
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    color=color, fontsize=7, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title("TASK-LEVEL HEATMAP — Pass rate per model × task", loc="left")

    path = out / "chart_task_heatmap.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ {path}")


# ── Master hero chart ────────────────────────────────────────

def chart_master(df: pd.DataFrame, lb: pd.DataFrame, fr: pd.DataFrame, out: Path):
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(PALETTE["bg"])
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Panel 1: Overall leaderboard
    overall = lb[lb["category"] == "OVERALL"].sort_values("avg_pass_rate", ascending=True)
    colors = [MODEL_COLORS.get(m, PALETTE["blue"]) for m in overall["model"]]
    bars = ax1.barh(overall["model"], overall["avg_pass_rate"], color=colors, height=0.55, alpha=0.9)
    for bar, val in zip(bars, overall["avg_pass_rate"]):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:.0%}", va="center", color=PALETTE["text"], fontsize=8)
    ax1.set_xlim(0, 1.2); ax1.set_xlabel("Pass Rate")
    ax1.set_title("LEADERBOARD", loc="left", fontsize=9)
    ax1.grid(axis="x", alpha=0.3)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # Panel 2: By category
    cats = lb[lb["category"] != "OVERALL"]
    cat_pivot = cats.pivot_table(index="category", columns="model",
                                 values="avg_pass_rate", aggfunc="mean").fillna(0)
    cat_pivot.index = [CATEGORY_LABELS.get(c, c) for c in cat_pivot.index]
    x2 = np.arange(len(cat_pivot.index))
    n2 = len(cat_pivot.columns)
    w2 = 0.7 / max(n2, 1)
    for i, model in enumerate(cat_pivot.columns):
        offset = (i - n2/2) * w2 + w2/2
        ax2.bar(x2 + offset, cat_pivot[model], width=w2*0.9,
                color=MODEL_COLORS.get(model, PALETTE["blue"]), alpha=0.85, label=model)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(cat_pivot.index, rotation=10, ha="right", fontsize=8)
    ax2.set_ylim(0, 1.15); ax2.set_ylabel("Pass Rate")
    ax2.set_title("BY CATEGORY", loc="left", fontsize=9)
    ax2.legend(framealpha=0, fontsize=7, ncol=2)
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    # Panel 3: By difficulty
    diff_order = ["easy","medium","hard"]
    diff_pivot = df.groupby(["difficulty","model"])["pass_rate"].mean().unstack(fill_value=0)
    diff_pivot = diff_pivot.reindex([d for d in diff_order if d in diff_pivot.index])
    x3 = np.arange(len(diff_pivot.index))
    n3 = len(diff_pivot.columns)
    w3 = 0.7 / max(n3, 1)
    for i, model in enumerate(diff_pivot.columns):
        offset = (i - n3/2) * w3 + w3/2
        ax3.bar(x3 + offset, diff_pivot[model], width=w3*0.9,
                color=MODEL_COLORS.get(model, PALETTE["blue"]), alpha=0.85, label=model)
    ax3.set_xticks(x3)
    ax3.set_xticklabels([d.upper() for d in diff_pivot.index])
    ax3.set_ylim(0, 1.15); ax3.set_ylabel("Pass Rate")
    ax3.set_title("BY DIFFICULTY — Easy → Hard degradation", loc="left", fontsize=9)
    ax3.grid(axis="y", alpha=0.3)
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    # Panel 4: Failure mode stacked
    if not fr.empty:
        modes = list(FAILURE_COLORS.keys())
        models_list = fr["model"].unique()
        matrix = pd.DataFrame(index=models_list, columns=modes, data=0.0)
        for _, row in fr.iterrows():
            if row["failure_mode"] in modes:
                matrix.loc[row["model"], row["failure_mode"]] = row["count"]
        bottom = np.zeros(len(models_list))
        x4 = np.arange(len(models_list))
        for mode in modes:
            vals = matrix[mode].values.astype(float)
            ax4.bar(x4, vals, bottom=bottom,
                    color=FAILURE_COLORS.get(mode, PALETTE["muted"]),
                    alpha=0.85, label=mode, width=0.55)
            bottom += vals
        ax4.set_xticks(x4)
        ax4.set_xticklabels(models_list, rotation=20, ha="right", fontsize=7)
        ax4.set_ylabel("Failure Count")
        ax4.set_title("FAILURE MODE BREAKDOWN", loc="left", fontsize=9)
        ax4.legend(framealpha=0, fontsize=7)
        ax4.grid(axis="y", alpha=0.3)
        ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

    fig.suptitle("INSTRUCTION FOLLOWING BENCHMARK — Results Overview",
                 fontsize=13, fontweight="bold", color=PALETTE["text"], y=0.99)

    path = out / "charts.png"
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"], dpi=150)
    plt.close()
    print(f"  ✓ {path}  ← README hero")


# ── Entry point ──────────────────────────────────────────────

def generate_all_charts(results_path=None):
    if not (RESULTS_DIR / "results.csv").exists():
        raise FileNotFoundError("Run run_experiment.py first.")
    setup_style()
    df, lb, fr = load_data()
    print(f"\n Generating charts...")
    print(f"  {len(df)} rows | {df['model'].nunique()} models | {df['task_id'].nunique()} tasks\n")

    chart_leaderboard(lb, RESULTS_DIR)
    chart_by_category(lb, RESULTS_DIR)
    chart_by_difficulty(df, RESULTS_DIR)
    chart_failure_modes(fr, RESULTS_DIR)
    chart_task_heatmap(df, RESULTS_DIR)
    chart_master(df, lb, fr, RESULTS_DIR)

    print(f"\n All charts saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    generate_all_charts()
