"""
update_findings.py
==================
Document Intelligence System — Auto-populate findings
Project: P9 · prompt-engineering-lab
"""

import json
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results")
README_PATH = Path("README.md")


def load():
    bdf = pd.read_csv(RESULTS_DIR/"benchmark_results.csv") if (RESULTS_DIR/"benchmark_results.csv").exists() else pd.DataFrame()
    with open(RESULTS_DIR/"pipeline_results.json") as f:
        results = json.load(f)
    return bdf, results


def build_accuracy_table(bdf):
    if bdf.empty:
        return "| Document | Classification | Entity Recall | Date Recall | QA Accuracy | Composite |\n|----------|---------------|---------------|-------------|-------------|-----------|"
    rows = [
        "| Document | Classification | Entity Recall | Date Recall | QA Accuracy | Composite |",
        "|----------|---------------|---------------|-------------|-------------|-----------|",
    ]
    for _, row in bdf.iterrows():
        rows.append(
            f"| {row['doc_id'].replace('_',' ')[:25]} "
            f"| {row.get('classification',0):.0%} "
            f"| {row.get('entity_recall',0):.0%} "
            f"| {row.get('date_recall',0):.0%} "
            f"| {row.get('qa_accuracy',0):.0%} "
            f"| {row.get('composite',0):.0%} |"
        )
    avg = bdf[["classification","entity_recall","date_recall","qa_accuracy","composite"]].mean()
    rows.append(
        f"| **AVERAGE** "
        f"| **{avg.get('classification',0):.0%}** "
        f"| **{avg.get('entity_recall',0):.0%}** "
        f"| **{avg.get('date_recall',0):.0%}** "
        f"| **{avg.get('qa_accuracy',0):.0%}** "
        f"| **{avg.get('composite',0):.0%}** |"
    )
    return "\n".join(rows)


def update_readme(acc_table):
    content = README_PATH.read_text(encoding="utf-8")
    start = "| Document | Classification |"
    end   = "\n\n*Run"
    si = content.find(start); ei = content.find(end, si)
    if si != -1 and ei != -1:
        content = content[:si] + acc_table + content[ei:]
    README_PATH.write_text(content, encoding="utf-8")
    print("  README.md updated.")


def print_findings(bdf, results):
    avg = bdf[["classification","entity_recall","date_recall","qa_accuracy","composite"]].mean() if not bdf.empty else {}

    types = [r.get("classification",{}).get("document_type","?") for r in results]
    correct_types = sum(1 for r in results
                       if r.get("classification",{}).get("confidence",0) >= 0.7)

    total_facts  = sum(len(r.get("extraction",{}).get("key_facts",[])) for r in results)
    total_money  = sum(len(r.get("extraction",{}).get("monetary_values",[])) for r in results)
    total_chunks = sum(r.get("chunks",0) for r in results)

    lat_classify = sum(r.get("latency_s",{}).get("classify",0) for r in results)/max(len(results),1)
    lat_extract  = sum(r.get("latency_s",{}).get("extract",0)  for r in results)/max(len(results),1)
    lat_qa       = sum(r.get("latency_s",{}).get("qa",0)       for r in results)/max(len(results),1)

    print("\n" + "="*62)
    print("COPY THIS INTO NOTEBOOK KEY FINDINGS CELL:")
    print("="*62)
    print(f"""
## Key Findings

1. **Classification accuracy:** {avg.get('classification',0):.0%} ({correct_types}/{len(results)} documents correctly typed with ≥70% confidence)
2. **Entity extraction recall:** {avg.get('entity_recall',0):.0%} of known entities found
3. **Date extraction recall:** {avg.get('date_recall',0):.0%} of known dates found
4. **QA accuracy:** {avg.get('qa_accuracy',0):.0%} of ground-truth questions answered correctly
5. **Composite accuracy:** {avg.get('composite',0):.0%} overall
6. **Extraction volume:** {total_facts} key facts, {total_money} monetary values across {len(results)} docs
7. **Chunking:** {total_chunks} total chunks ({total_chunks//max(len(results),1)} avg per document)
8. **Avg latency:** classify={lat_classify:.1f}s · extract={lat_extract:.1f}s · QA={lat_qa:.1f}s per doc

---
*Demo: `python app.py` → http://127.0.0.1:7860*
""")


def main():
    print("Loading results...")
    bdf, results = load()
    print(f"  {len(results)} documents processed\n")
    print("Building README tables...")
    update_readme(build_accuracy_table(bdf))
    print_findings(bdf, results)
    print("\nDone.")

if __name__ == "__main__":
    main()
