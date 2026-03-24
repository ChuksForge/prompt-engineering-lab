"""
update_findings.py
==================
Hallucination Detection & Mitigation — Auto-populate findings
Project: P8 · prompt-engineering-lab
"""

from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results")
README_PATH = Path("README.md")


def load():
    metrics = pd.read_csv(RESULTS_DIR/"detector_metrics.csv")
    roc     = pd.read_csv(RESULTS_DIR/"roc_data.csv")
    det     = pd.read_csv(RESULTS_DIR/"detection_results.csv")
    mit_sum = pd.read_csv(RESULTS_DIR/"mitigation_summary.csv") if (RESULTS_DIR/"mitigation_summary.csv").exists() else pd.DataFrame()
    return metrics, roc, det, mit_sum


def build_detector_table(metrics, roc):
    auc_map = roc.groupby("detector")["auc"].first().to_dict()
    rows = [
        "| Detector | Precision | Recall | F1 | AUC | Accuracy |",
        "|----------|-----------|--------|----|-----|----------|",
    ]
    for _, row in metrics.sort_values("f1", ascending=False).iterrows():
        auc = auc_map.get(row["detector"], 0)
        rows.append(
            f"| {row['detector']} "
            f"| {row['precision']:.3f} "
            f"| {row['recall']:.3f} "
            f"| {row['f1']:.3f} "
            f"| {auc:.3f} "
            f"| {row['accuracy']:.3f} |"
        )
    return "\n".join(rows)


def build_mitigation_table(mit_sum):
    if mit_sum.empty:
        return "| Strategy | Success Rate | Avg Improvement |\n|----------|-------------|----------------|\n| — | — | — |"
    rows = [
        "| Strategy | Success Rate | Avg Improvement |",
        "|----------|-------------|----------------|",
    ]
    for _, row in mit_sum.sort_values("success_rate", ascending=False).iterrows():
        rows.append(f"| {row['strategy']} | {row['success_rate']:.1%} | {row['avg_improvement']:+.3f} |")
    return "\n".join(rows)


def update_readme(det_table, mit_table):
    content = README_PATH.read_text(encoding="utf-8")
    for start, end, replacement in [
        ("| Detector | Precision |", "\n\n*Run", det_table),
        ("| Strategy | Success Rate |", "\n\n---", mit_table),
    ]:
        si = content.find(start); ei = content.find(end, si)
        if si != -1 and ei != -1:
            content = content[:si] + replacement + content[ei:]
    README_PATH.write_text(content, encoding="utf-8")
    print("  README.md updated.")


def print_findings(metrics, roc, det, mit_sum):
    best_detector = metrics.sort_values("f1", ascending=False).iloc[0]
    auc_map = roc.groupby("detector")["auc"].first()
    best_auc_det = auc_map.idxmax(); best_auc = auc_map.max()

    # False positive rates
    fpr_at_50 = {}
    for det_name, grp in roc.groupby("detector"):
        near50 = grp.iloc[(grp["tpr"] - 0.5).abs().argsort()[:1]]
        fpr_at_50[det_name] = near50["fpr"].values[0] if not near50.empty else 0

    best_mit = ""
    if not mit_sum.empty:
        top_mit = mit_sum.sort_values("success_rate", ascending=False).iloc[0]
        best_mit = f"`{top_mit['strategy']}` — {top_mit['success_rate']:.0%} success rate"

    total_claims = len(det["claim_id"].unique()) if not det.empty else 0
    total_hals   = det[det["ground_truth_hal"]==True]["claim_id"].nunique() if not det.empty else 0

    print("\n" + "="*62)
    print("COPY THIS INTO NOTEBOOK KEY FINDINGS CELL:")
    print("="*62)
    print(f"""
## Key Findings

1. **Best detector (F1):** `{best_detector['detector']}` — P={best_detector['precision']:.3f} R={best_detector['recall']:.3f} F1={best_detector['f1']:.3f}
2. **Best detector (AUC):** `{best_auc_det}` — AUC={best_auc:.3f}
3. **False positive tradeoff:** At 50% recall, {', '.join(f'{d}={v:.1%} FPR' for d,v in fpr_at_50.items())}
4. **Best mitigation strategy:** {best_mit if best_mit else 'See results/mitigation_summary.csv'}
5. **Dataset:** {total_claims} claims, {total_hals} labeled hallucinations ({total_hals/total_claims:.0%} hallucination rate)
6. **Key insight:** [Fill in after reviewing chart_roc.png and chart_prf.png]

---
*Demo: `python app.py` → http://127.0.0.1:7860*
""")


def main():
    print("Loading results...")
    metrics, roc, det, mit_sum = load()
    print(f"  {len(metrics)} detectors | {det['claim_id'].nunique()} claims\n")
    print("Building README tables...")
    update_readme(build_detector_table(metrics, roc), build_mitigation_table(mit_sum))
    print_findings(metrics, roc, det, mit_sum)
    print("\nDone.")

if __name__ == "__main__":
    main()
