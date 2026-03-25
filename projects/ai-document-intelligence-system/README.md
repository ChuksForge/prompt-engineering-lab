# 📑 P9 — AI Document Intelligence System

> **End-to-end document processing pipeline: multi-format ingestion, classification, structured extraction, intelligent QA, and accuracy benchmarking**  
> Part of the [prompt-engineering-lab](../../README.md) portfolio · **Capstone Project**

---

## Overview

A complete document intelligence system that takes any document — contract, invoice, report, meeting minutes, financial statement — and extracts structured meaning from it, routes it by type, answers natural-language questions about it, and benchmarks accuracy against ground truth.

| | |
|---|---|
| **Formats** | TXT · PDF (pypdf) · DOCX (python-docx) · CSV · Markdown |
| **Pipeline** | Ingest → Classify → Extract → Chunk → Index → QA → Benchmark |
| **Documents** | 5 sample documents across 5 document types |
| **Models** | GPT-4o-mini · GPT-4o · Claude Haiku · Claude Sonnet 4.6 · Mistral 7B |
| **Metrics** | Classification accuracy · Entity recall · Date recall · QA accuracy |
| **Demo** | Gradio — upload or paste any document, instant pipeline results |

---

## Results

![Document Intelligence Results](results/charts.png)

### Accuracy Benchmark

| Document | Classification | Entity Recall | Date Recall | QA Accuracy | Composite |
|----------|---------------|---------------|-------------|-------------|-----------|
| contract service agreemen | 100% | 100% | 100% | 100% | 100% |
| financial statement novat | 100% | 100% | 50% | 100% | 90% |
| invoice design services | 100% | 100% | 100% | 80% | 94% |
| meeting minutes q1 roadma | 100% | 100% | 100% | 80% | 94% |
| research report ai produc | 100% | 100% | 100% | 60% | 88% |
| **AVERAGE** | **100%** | **100%** | **90%** | **84%** | **93%** |

*Run `python update_findings.py` after the pipeline to populate.*

---

## Project Structure

```
document-intelligence/
├── app.py               ← Gradio demo (python app.py → localhost:7860)
├── pipeline.py          ← Full pipeline orchestrator + benchmark runner
├── ingestion.py         ← Multi-format document reader (TXT/PDF/DOCX/CSV)
├── chunker.py           ← Sentence-aware chunker + TF-IDF indexer
├── intelligence.py      ← Classifier + Extractor + DocumentQA (LLM-powered)
├── visualize.py         ← Chart generator
├── update_findings.py   ← Auto-populate README + findings
├── experiment.ipynb     ← Analysis notebook
├── data/
│   ├── documents/       ← 5 sample documents
│   │   ├── contract_service_agreement.txt
│   │   ├── invoice_design_services.txt
│   │   ├── research_report_ai_productivity.txt
│   │   ├── meeting_minutes_q1_roadmap.txt
│   │   └── financial_statement_novatech.txt
│   └── ground_truth/
│       └── ground_truth.json    ← Expected extractions + QA pairs
└── results/
    ├── pipeline_results.json
    ├── benchmark_results.csv
    └── charts.png
```

---

## Quick Start

```bash
pip install -r requirements.txt

export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENROUTER_API_KEY="sk-or-..."

# Quick test (2 documents)
python pipeline.py --quick --models openai

# Full pipeline (all 5 documents)
python pipeline.py --models openai

# With optional ML embeddings
pip install sentence-transformers
python pipeline.py --use-embeddings

# Charts + README
python visualize.py
python update_findings.py

# Live demo
python app.py
```

---

## CLI Options

```
python pipeline.py [options]

  --models           openai,anthropic,openrouter
  --doc              contract_service_agreement.txt  (single doc)
  --quick            2 documents only
  --no-benchmark     skip accuracy scoring
  --use-embeddings   use sentence-transformers for retrieval
```

---

## Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| **Ingest** | `ingestion.py` | Read TXT/PDF/DOCX/CSV, extract clean text |
| **Classify** | `intelligence.py` | LLM document type classification + routing tags |
| **Extract** | `intelligence.py` | Entities, dates, monetary values, key facts, action items |
| **Chunk** | `chunker.py` | Sentence-aware splitting with section detection |
| **Index** | `chunker.py` | TF-IDF cosine index (optional: dense embeddings) |
| **QA** | `intelligence.py` | RAG-based question answering with citations |
| **Benchmark** | `pipeline.py` | Accuracy scoring against ground truth |

---

## Document Types Supported

| Type | Examples | Key Extractions |
|------|----------|----------------|
| `contract` | Service agreements, NDAs, leases | Parties, dates, fees, terms, obligations |
| `invoice` | Bills, purchase orders | Line items, totals, payment terms, parties |
| `research_report` | Academic papers, analysis | Authors, methodology, findings, metrics |
| `meeting_minutes` | Board/team meetings | Attendees, decisions, action items, dates |
| `financial_statement` | P&L, balance sheets | Revenue, expenses, ratios, period |

---

## Optional Enhancements

```bash
# Better PDF extraction
pip install pypdf

# Better DOCX extraction
pip install python-docx

# Dense embeddings for retrieval (higher accuracy, ~400MB)
pip install sentence-transformers
```

All optional — pipeline works with zero ML dependencies using TF-IDF fallbacks.

---

## Related Projects

- **P5:** [Grounded QA](../grounded-qa/) — same RAG retrieval pattern
- **P8:** [Hallucination Detection](../hallucination-detection/) — can be integrated post-extraction
- **P4:** [Prompt Testing Framework](../prompt-testing-framework/) — `promptlab` can wrap extraction prompts

---

*prompt-engineering-lab / projects / document-intelligence · Capstone*
