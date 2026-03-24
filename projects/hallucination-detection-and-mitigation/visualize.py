"""
visualize.py
============
Hallucination Detection & Mitigation — Chart Generator
Project: P8 · prompt-engineering-lab

Charts:
  1. Detector comparison (P/R/F1 bar chart)
  2. ROC curves per detector
  3. Hallucination type detection rate heatmap
  4. Mitigation success rate by strategy
  5. Confidence distribution (detected vs missed)
  6. Master 4-panel hero → results/charts.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

RESULTS_DIR = Path("results")

PALETTE = {
    "bg":"#0f1117","surface":"#161820","border":"#1e2130",
    "text":"#f0f2f8","muted":"#5a6080","accent":"#e8ff47",
    "blue":"#47c8ff","purple":"#b847ff","orange":"#ff8c47",
    "green":"#47ffb2","red":"#ff4776","pink":"#ff47c8",
}
DETECTOR_COLORS = {
    "rule_based":     PALETTE["blue"],
    "llm_judge":      PALETTE["accent"],
    "entailment":     PALETTE["purple"],
    "entailment_nli": PALETTE["purple"],
    "entailment_cosine": PALETTE["orange"],
}
STRATEGY_COLORS = {
    "grounded_rewrite":  PALETTE["green"],
    "self_critique":     PALETTE["blue"],
    "citation_enforced": PALETTE["accent"],
}

def setup_style():
    plt.rcParams.update({
        "figure.facecolor":PALETTE["bg"],"axes.facecolor":PALETTE["surface"],
        "axes.edgecolor":PALETTE["border"],"axes.labelcolor":PALETTE["text"],
        "axes.titlecolor":PALETTE["text"],"xtick.color":PALETTE["muted"],
        "ytick.color":PALETTE["muted"],"text.color":PALETTE["text"],
        "grid.color":PALETTE["border"],"grid.linewidth":0.5,
        "font.family":"monospace","font.size":9,"axes.titlesize":10,
        "axes.titleweight":"bold","figure.dpi":150,
    })

def load_data():
    metrics = pd.read_csv(RESULTS_DIR/"detector_metrics.csv")
    roc     = pd.read_csv(RESULTS_DIR/"roc_data.csv")
    det     = pd.read_csv(RESULTS_DIR/"detection_results.csv")
    mit_sum = pd.read_csv(RESULTS_DIR/"mitigation_summary.csv") if (RESULTS_DIR/"mitigation_summary.csv").exists() else pd.DataFrame()
    mit_res = pd.read_csv(RESULTS_DIR/"mitigation_results.csv") if (RESULTS_DIR/"mitigation_results.csv").exists() else pd.DataFrame()
    return metrics, roc, det, mit_sum, mit_res


# ── Chart 1: P/R/F1 bar ─────────────────────────────────────

def chart_prf(metrics, out):
    if metrics.empty:
        return
    detectors = metrics["detector"].tolist()
    x = np.arange(len(detectors))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    metric_data = {
        "Precision": (metrics["precision"].values, PALETTE["blue"]),
        "Recall":    (metrics["recall"].values,    PALETTE["green"]),
        "F1":        (metrics["f1"].values,        PALETTE["accent"]),
    }
    for i, (label, (vals, color)) in enumerate(metric_data.items()):
        offset = (i - 1) * w
        bars = ax.bar(x + offset, vals, width=w*0.9, color=color, alpha=0.85, label=label)
        for bar, val in zip(bars, vals):
            if val > 0.05:
                ax.text(bar.get_x()+bar.get_width()/2, val+0.01, f"{val:.2f}",
                        ha="center", fontsize=7, color=PALETTE["text"])
    ax.set_xticks(x); ax.set_xticklabels(detectors)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
    ax.set_title("DETECTOR COMPARISON — Precision / Recall / F1", loc="left")
    ax.legend(framealpha=0, fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = out/"chart_prf.png"
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"]); plt.close()
    print(f"  ✓ {path}")


# ── Chart 2: ROC curves ──────────────────────────────────────

def chart_roc(roc, out):
    if roc.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0,1],[0,1], color=PALETTE["muted"], linestyle="--", linewidth=0.8, alpha=0.5, label="Random")
    for det, grp in roc.groupby("detector"):
        color = DETECTOR_COLORS.get(det, PALETTE["blue"])
        auc   = grp["auc"].iloc[0]
        ax.plot(grp["fpr"], grp["tpr"], color=color, linewidth=2,
                label=f"{det} (AUC={auc:.3f})")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC CURVES — Per Detector", loc="left")
    ax.legend(framealpha=0, fontsize=8); ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = out/"chart_roc.png"
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"]); plt.close()
    print(f"  ✓ {path}")


# ── Chart 3: Mitigation success ──────────────────────────────

def chart_mitigation(mit_sum, out):
    if mit_sum.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(mit_sum)); w = 0.35
    colors1 = [STRATEGY_COLORS.get(s, PALETTE["blue"]) for s in mit_sum["strategy"]]
    bars1 = ax.bar(x - w/2, mit_sum["success_rate"], width=w*0.9,
                   color=colors1, alpha=0.85, label="Success Rate")
    bars2 = ax.bar(x + w/2, mit_sum["avg_improvement"], width=w*0.9,
                   color=PALETTE["muted"], alpha=0.6, label="Avg Improvement")
    for bar, val in zip(bars1, mit_sum["success_rate"]):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.01, f"{val:.0%}",
                ha="center", fontsize=8, color=PALETTE["text"])
    ax.set_xticks(x); ax.set_xticklabels(mit_sum["strategy"], rotation=10, ha="right")
    ax.set_ylim(0, 1.2); ax.set_ylabel("Score / Rate")
    ax.set_title("MITIGATION SUCCESS — Strategy Comparison", loc="left")
    ax.legend(framealpha=0, fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = out/"chart_mitigation.png"
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"]); plt.close()
    print(f"  ✓ {path}")


# ── Chart 4: Confidence distribution ────────────────────────

def chart_confidence(det, out):
    if det.empty or "ground_truth_hal" not in det.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for detector, grp in det.groupby("detector"):
        color = DETECTOR_COLORS.get(detector, PALETTE["blue"])
        hal   = grp[grp["ground_truth_hal"] == True]["confidence"]
        clean = grp[grp["ground_truth_hal"] == False]["confidence"]
        ax.scatter(range(len(hal)), sorted(hal), color=color, alpha=0.7, s=40,
                   marker="x", label=f"{detector} — actual HAL")
        ax.scatter(range(len(clean)), sorted(clean), color=color, alpha=0.3, s=20,
                   marker="o", label=f"{detector} — actual CLEAN")
    ax.axhline(0.5, color=PALETTE["muted"], linestyle="--", linewidth=0.8, alpha=0.6,
               label="decision threshold (0.5)")
    ax.set_xlabel("Claim (sorted by confidence)"); ax.set_ylabel("Detector Confidence")
    ax.set_title("CONFIDENCE DISTRIBUTION — Hallucinated vs Clean Claims", loc="left")
    ax.legend(framealpha=0, fontsize=7, ncol=2); ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = out/"chart_confidence.png"
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"]); plt.close()
    print(f"  ✓ {path}")


# ── Master hero chart ────────────────────────────────────────

def chart_master(metrics, roc, mit_sum, det, out):
    fig = plt.figure(figsize=(16,10))
    fig.patch.set_facecolor(PALETTE["bg"])
    gs = GridSpec(2,2,figure=fig,hspace=0.45,wspace=0.35)
    ax1,ax2 = fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1])
    ax3,ax4 = fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])

    # Panel 1: P/R/F1
    if not metrics.empty:
        detectors = metrics["detector"].tolist()
        x = np.arange(len(detectors)); w = 0.25
        for i, (label, col) in enumerate([("Precision",PALETTE["blue"]),("Recall",PALETTE["green"]),("F1",PALETTE["accent"])]):
            vals = metrics[label.lower()].values
            offset = (i-1)*w
            ax1.bar(x+offset, vals, width=w*0.9, color=col, alpha=0.85, label=label)
        ax1.set_xticks(x); ax1.set_xticklabels(detectors, fontsize=8)
        ax1.set_ylim(0,1.15); ax1.set_title("PRECISION / RECALL / F1",loc="left",fontsize=9)
        ax1.legend(framealpha=0,fontsize=7); ax1.grid(axis="y",alpha=0.3)
        ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # Panel 2: ROC
    if not roc.empty:
        ax2.plot([0,1],[0,1],color=PALETTE["muted"],linestyle="--",linewidth=0.8,alpha=0.5)
        for det_name, grp in roc.groupby("detector"):
            color = DETECTOR_COLORS.get(det_name, PALETTE["blue"])
            auc = grp["auc"].iloc[0]
            ax2.plot(grp["fpr"],grp["tpr"],color=color,linewidth=2,label=f"{det_name} AUC={auc:.3f}")
        ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
        ax2.set_title("ROC CURVES",loc="left",fontsize=9)
        ax2.legend(framealpha=0,fontsize=7); ax2.grid(alpha=0.3)
        ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    # Panel 3: Mitigation
    if not mit_sum.empty:
        x3 = np.arange(len(mit_sum))
        colors3 = [STRATEGY_COLORS.get(s, PALETTE["blue"]) for s in mit_sum["strategy"]]
        bars3 = ax3.bar(x3, mit_sum["success_rate"], color=colors3, alpha=0.85, width=0.55)
        for bar, val in zip(bars3, mit_sum["success_rate"]):
            ax3.text(bar.get_x()+bar.get_width()/2, val+0.01, f"{val:.0%}",
                     ha="center", fontsize=8, color=PALETTE["text"])
        ax3.set_xticks(x3); ax3.set_xticklabels(mit_sum["strategy"], rotation=15, ha="right", fontsize=8)
        ax3.set_ylim(0,1.2); ax3.set_title("MITIGATION SUCCESS RATE",loc="left",fontsize=9)
        ax3.grid(axis="y",alpha=0.3)
        ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    # Panel 4: Correct vs incorrect per detector
    if not det.empty and "correct" in det.columns:
        det_acc = det.groupby("detector")["correct"].mean().sort_values(ascending=True)
        colors4 = [DETECTOR_COLORS.get(d, PALETTE["blue"]) for d in det_acc.index]
        bars4 = ax4.barh(det_acc.index, det_acc.values, color=colors4, height=0.55, alpha=0.9)
        for bar, val in zip(bars4, det_acc.values):
            ax4.text(val+0.01, bar.get_y()+bar.get_height()/2, f"{val:.0%}",
                     va="center", fontsize=8, color=PALETTE["text"])
        ax4.set_xlim(0,1.2); ax4.set_xlabel("Accuracy")
        ax4.set_title("DETECTOR ACCURACY",loc="left",fontsize=9)
        ax4.grid(axis="x",alpha=0.3)
        ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

    fig.suptitle("HALLUCINATION DETECTION & MITIGATION — Results Overview",
                 fontsize=13,fontweight="bold",color=PALETTE["text"],y=0.99)
    path = out/"charts.png"
    plt.savefig(path,bbox_inches="tight",facecolor=PALETTE["bg"],dpi=150)
    plt.close(); print(f"  ✓ {path}  ← README hero")


def generate_all_charts():
    if not (RESULTS_DIR/"detector_metrics.csv").exists():
        raise FileNotFoundError("Run pipeline.py first.")
    setup_style()
    metrics, roc, det, mit_sum, mit_res = load_data()
    print(f"\n Generating charts...\n")
    chart_prf(metrics, RESULTS_DIR)
    chart_roc(roc, RESULTS_DIR)
    if not mit_sum.empty: chart_mitigation(mit_sum, RESULTS_DIR)
    chart_confidence(det, RESULTS_DIR)
    chart_master(metrics, roc, mit_sum, det, RESULTS_DIR)
    print(f"\n Charts saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    generate_all_charts()
