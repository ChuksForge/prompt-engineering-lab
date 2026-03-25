"""
visualize.py
============
Document Intelligence System — Chart Generator
Project: P9 · prompt-engineering-lab
"""

from pathlib import Path
import json
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
    "green":"#47ffb2","red":"#ff4776",
}
DOC_COLORS = {
    "contract_service_agreement":     PALETTE["blue"],
    "invoice_design_services":        PALETTE["accent"],
    "research_report_ai_productivity":PALETTE["purple"],
    "meeting_minutes_q1_roadmap":     PALETTE["green"],
    "financial_statement_novatech":   PALETTE["orange"],
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
    bdf = pd.read_csv(RESULTS_DIR/"benchmark_results.csv") if (RESULTS_DIR/"benchmark_results.csv").exists() else pd.DataFrame()
    with open(RESULTS_DIR/"pipeline_results.json") as f:
        results = json.load(f)
    return bdf, results


def chart_master(bdf, results, out):
    fig = plt.figure(figsize=(16,10))
    fig.patch.set_facecolor(PALETTE["bg"])
    gs = GridSpec(2,2,figure=fig,hspace=0.45,wspace=0.35)
    ax1,ax2 = fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1])
    ax3,ax4 = fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])

    # Panel 1: Accuracy per document
    if not bdf.empty:
        metrics = ["classification","entity_recall","date_recall","qa_accuracy","composite"]
        available = [m for m in metrics if m in bdf.columns]
        x = np.arange(len(bdf)); w = 0.7/len(available)
        metric_colors = [PALETTE["blue"],PALETTE["green"],PALETTE["accent"],PALETTE["purple"],PALETTE["orange"]]
        for i,metric in enumerate(available):
            offset = (i-len(available)/2)*w + w/2
            ax1.bar(x+offset, bdf[metric], width=w*0.9,
                    color=metric_colors[i%len(metric_colors)], alpha=0.85, label=metric)
        ax1.set_xticks(x)
        short_names = [d.replace("_"," ")[:15] for d in bdf["doc_id"]]
        ax1.set_xticklabels(short_names, rotation=20, ha="right", fontsize=7)
        ax1.set_ylim(0,1.2); ax1.set_ylabel("Accuracy Score")
        ax1.set_title("ACCURACY PER DOCUMENT",loc="left",fontsize=9)
        ax1.legend(framealpha=0,fontsize=7,ncol=2)
        ax1.grid(axis="y",alpha=0.3)
        ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # Panel 2: Document type classification results
    types = [r.get("classification",{}).get("document_type","unknown") for r in results]
    type_counts = {}
    for t in types:
        type_counts[t] = type_counts.get(t,0) + 1
    if type_counts:
        tc_items = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        labels_tc, vals_tc = zip(*tc_items)
        colors_tc = [PALETTE["blue"],PALETTE["accent"],PALETTE["purple"],PALETTE["green"],PALETTE["orange"]]
        ax2.barh(labels_tc, vals_tc, color=colors_tc[:len(vals_tc)], height=0.55, alpha=0.9)
        ax2.set_xlabel("Count"); ax2.set_title("DOCUMENT TYPES DETECTED",loc="left",fontsize=9)
        ax2.grid(axis="x",alpha=0.3)
        ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    # Panel 3: Metric comparison (avg)
    if not bdf.empty:
        metrics_avg = bdf[["classification","entity_recall","date_recall","qa_accuracy","composite"]].mean()
        metric_colors2 = [PALETTE["blue"],PALETTE["green"],PALETTE["accent"],PALETTE["purple"],PALETTE["orange"]]
        bars3 = ax3.bar(metrics_avg.index, metrics_avg.values,
                        color=metric_colors2[:len(metrics_avg)], alpha=0.85, width=0.6)
        for bar,val in zip(bars3,metrics_avg.values):
            ax3.text(bar.get_x()+bar.get_width()/2, val+0.01, f"{val:.2f}",
                     ha="center",fontsize=8,color=PALETTE["text"])
        ax3.set_ylim(0,1.2); ax3.set_ylabel("Avg Score")
        ax3.set_title("AVERAGE ACCURACY BY METRIC",loc="left",fontsize=9)
        ax3.set_xticklabels([m.replace("_","\n") for m in metrics_avg.index],fontsize=8)
        ax3.grid(axis="y",alpha=0.3)
        ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    # Panel 4: Pipeline latency
    latencies = []
    for r in results:
        lat = r.get("latency_s",{})
        total = sum(lat.values())
        latencies.append({"doc": r.get("doc_id","")[:15], "classify": lat.get("classify",0),
                          "extract": lat.get("extract",0), "index": lat.get("index",0), "qa": lat.get("qa",0)})
    if latencies:
        ldf = pd.DataFrame(latencies)
        x4 = np.arange(len(ldf)); w4=0.2
        stage_colors = [PALETTE["blue"],PALETTE["purple"],PALETTE["green"],PALETTE["accent"]]
        for i,stage in enumerate(["classify","extract","index","qa"]):
            if stage in ldf.columns:
                ax4.bar(x4+i*w4, ldf[stage], width=w4*0.9,
                        color=stage_colors[i], alpha=0.85, label=stage)
        ax4.set_xticks(x4+w4); ax4.set_xticklabels(ldf["doc"], rotation=20, ha="right", fontsize=7)
        ax4.set_ylabel("Latency (s)"); ax4.set_title("PIPELINE LATENCY BY STAGE",loc="left",fontsize=9)
        ax4.legend(framealpha=0,fontsize=7)
        ax4.grid(axis="y",alpha=0.3)
        ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

    fig.suptitle("AI DOCUMENT INTELLIGENCE SYSTEM — Results Overview",
                 fontsize=13,fontweight="bold",color=PALETTE["text"],y=0.99)
    path = out/"charts.png"
    plt.savefig(path,bbox_inches="tight",facecolor=PALETTE["bg"],dpi=150)
    plt.close(); print(f"  ✓ {path}")


def generate_all_charts():
    if not (RESULTS_DIR/"pipeline_results.json").exists():
        raise FileNotFoundError("Run pipeline.py first.")
    setup_style()
    bdf, results = load_data()
    print(f"\n Generating charts ({len(results)} documents)...\n")
    chart_master(bdf, results, RESULTS_DIR)
    print(f"\n Charts saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    generate_all_charts()
