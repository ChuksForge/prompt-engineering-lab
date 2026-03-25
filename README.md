# Prompt Engineering Lab

> A research-grade portfolio of prompt engineering systems — from evaluation benchmarks to production AI pipelines.

---

## About

This repository documents a complete prompt engineering portfolio built across 9 projects, organized into three tiers of increasing complexity. Each project is fully reproducible: all prompts, evaluation code, datasets, results, and charts are committed.

The portfolio demonstrates the full spectrum of prompt engineering work — systematic measurement, failure analysis, cost optimization, hallucination mitigation, and production-ready tooling.

**Stack:** Python · OpenAI · Anthropic · OpenRouter · Pandas · Matplotlib · Gradio · Streamlit · Jupyter

---

## Portfolio at a Glance

| # | Project | Tier | Key Output | Signal |
|---|---------|------|-----------|--------|
| P1 | [Summarization Benchmark](#p1--summarization-benchmark) | Foundation | Multi-model ROUGE leaderboard | Eval methodology |
| P2 | [Style Transfer Prompts](#p2--style-transfer-prompts) | Foundation | 10-style gallery + tone metrics | Range & creativity |
| P3 | [Instruction Following Benchmark](#p3--instruction-following-benchmark) | Foundation | Programmatic rubric + failure taxonomy | Rigor |
| P4 | [Prompt Testing Framework](#p4--prompt-testing-framework-promptlab) | Foundation | Installable Python library + CLI | Engineering depth |
| P5 | [Grounded QA (Anti-Hallucination)](#p5--grounded-qa-anti-hallucination) | Applied | RAG pipeline + hallucination rate curves | Enterprise signal |
| P6 | [AI Email Summarizer](#p6--ai-email-summarizer) | Applied | Tone-aware summarizer + Gradio demo | Product thinking |
| P7 | [LLM Prompt Benchmark System](#p7--llm-prompt-benchmark-system) | Advanced | Cost-per-quality leaderboard + Streamlit dashboard | Research scale |
| P8 | [Hallucination Detection & Mitigation](#p8--hallucination-detection--mitigation) | Advanced | 3-detector pipeline + precision/recall curves | #1 enterprise concern |
| P9 | [AI Document Intelligence System](#p9--ai-document-intelligence-system) | Advanced | End-to-end doc pipeline + accuracy benchmark | Capstone complexity |

---

## Repository Structure

```
prompt-engineering-lab/
│
├── README.md
│
├── projects/
│   ├── summarization-benchmark/        ← P1
│   ├── style-transfer-prompts/         ← P2
│   ├── instruction-following/          ← P3
│   ├── prompt-testing-framework/       ← P4  (pip installable: promptlab)
│   ├── grounded-qa/                    ← P5
│   ├── email-summarizer/               ← P6
│   ├── prompt-benchmark-system/        ← P7
│   ├── hallucination-detection/        ← P8
│   └── document-intelligence/          ← P9
│
├── notebooks/
│   └── prompt_testing.ipynb            ← Shared experimentation notebook
│
├── utils/
│   └── evaluation_tools.py             ← Shared evaluation utilities
│
└── results/                            ← Aggregate charts and reports
```

Each project folder contains: `README.md` · `experiment.ipynb` · `run_*.py` · `evaluation.py` · `prompts/` · `data/` · `results/`

---

## Skills Demonstrated

**Prompt Engineering**
- Zero-shot, few-shot, chain-of-thought, role prompting, contrastive DO/DON'T instructions
- Structured extraction prompts, citation-enforced grounding, tone-matched generation
- Systematic A/B comparison and prompt regression testing

**Evaluation & Measurement**
- ROUGE-1/2/L, BERTScore, Flesch-Kincaid, compression ratio
- Programmatic rubric scoring (16 constraint types, zero LLM cost)
- Precision, recall, F1, AUC/ROC curves for detection tasks
- Cost-per-quality metric (quality score ÷ API cost in USD)

**Hallucination & Grounding**
- 3-strategy detection pipeline: rule-based, LLM judge, NLI entailment
- 3-strategy mitigation: grounded rewrite, self-critique, citation enforcement
- RAG pipeline with TF-IDF retrieval (optional: sentence-transformers upgrade)

**Infrastructure & Tooling**
- `promptlab` — installable Python library for batch eval, A/B testing, regression baselines
- Multi-provider unified client (OpenAI / Anthropic / OpenRouter)
- Streamlit and Gradio demos for 3 projects
- `update_findings.py` pattern: auto-populates README tables from results CSVs

**APIs & Models Benchmarked**
- GPT-4o, GPT-4o-mini, Claude Sonnet 4.6, Claude Haiku, Mistral small creative, Llama 3 8B, Gemini Flash

---

## Project Summaries

### P1 · Summarization Benchmark

**Overview:** Multi-model, multi-prompt summarization quality benchmark across 10 prompt strategies and 5 domains.

**Key features:** ROUGE/BERTScore leaderboard comparing 6 models. Cost-per-quality chart showing which model gives the best score per dollar.

**Highlights:** The composite score (0.2×ROUGE-1 + 0.2×ROUGE-2 + 0.2×ROUGE-L + 0.4×BERTScore) weights semantic similarity over lexical overlap — a more defensible metric than raw ROUGE.

📁 [`projects/summarization-benchmark/`](projects/summarization-benchmark/)

---

### P2 · Style Transfer Prompts

**Overview:** 10 writing styles × 3 prompt strategies (direct, role-based, contrastive) × 6 models. Measures how much each strategy actually shifts tone, formality, and readability.

**Key features:** Side-by-side HTML gallery of all 300 outputs. Formality shift heatmap. FK grade level per style.

**Highlights:** `tone_detector.py` classifies source tone and auto-selects the matching prompt strategy. `update_findings.py` auto-populates every table.

📁 [`projects/style-transfer-prompts/`](projects/style-transfer-prompts/)

---

### P3 · Instruction Following Benchmark

**Overview:** 18 tasks across 3 categories (multi-step, tone/persona, negation handling) scored by a programmatic constraint rubric — no LLM judge needed.

**Key features:** Per-model failure taxonomy. Task heatmap showing exactly which tasks and models fail. Full compliance % vs pass rate distinction.

**Highlights:** 16 constraint types (word_absent, exact_phrase, numbered_list, allocation_sum, sentence_not_starts_with, etc.) that produce machine-verifiable pass/fail with human-readable explanations.

📁 [`projects/instruction-following/`](projects/instruction-following/)

---

### P4 · Prompt Testing Framework (`promptlab`)

**Overview:** Installable Python library (`pip install -e .`) for batch evaluation, A/B comparison with statistical significance testing, and regression baseline tracking.

**Key features:** `promptlab` package with CLI (`promptlab run`, `promptlab ab`, `promptlab regression`). YAML test file format. Baseline JSON snapshots committed to git.

**Highlights:** The infrastructure layer that powers every other project. Cross-referenced by P1–P3 and extended by P7–P8. This is the portfolio's strongest differentiation signal — you build tools, not just use them.

📁 [`projects/prompt-testing-framework/`](projects/prompt-testing-framework/)

---

### P5 · Grounded QA (Anti-Hallucination)

**Overview:** Measures hallucination rates across 3 conditions (ungrounded, grounded, cited) on 15 benchmark questions including 5 deliberately unanswerable traps.

**Key features:** Hallucination rate comparison chart. Context leak detection. RAG pipeline over 3 domain documents using TF-IDF retrieval. Citation validity scoring.

**Highlights:** The unanswerable traps are designed to catch specific hallucination patterns (fabricated numbers, context leakage, invented entities) and labelled with the exact model failure mode they're testing.

📁 [`projects/grounded-qa/`](projects/grounded-qa/)

---

### P6 · AI Email Summarizer

**Overview:** End-to-end email summarization product with tone detection, 5 prompt strategies, thread support, latency benchmarking, and a live Gradio demo.

**Key features:** Tone-matched summarization (auto-selects formal/casual/urgent/negative/positive prompt). Quality-per-second metric. 20-email benchmark with reference summaries.

**Highlights:** `tone_detector.py` is a zero-dependency 5-class tone classifier that drives automatic strategy selection — detect → adapt → evaluate is a complete loop.

📁 [`projects/email-summarizer/`](projects/email-summarizer/)

---

### P7 · LLM Prompt Benchmark System

**Overview:** Research-grade multi-task (summarization, QA, reasoning, coding), multi-model benchmark with cost-per-quality analysis across 60 prompt variants.

**Key features:** Cost-efficiency leaderboard (quality per dollar). Streamlit dashboard with 5 tabs: leaderboard, cost analysis, task breakdown, strategy comparison, raw explorer.

**Highlights:** `costs.py` maintains a token pricing table for all 6 models and computes `quality_per_dollar` on every result — directly answering the most practical production question: "is the expensive model worth it?"

📁 [`projects/prompt-benchmark-system/`](projects/prompt-benchmark-system/)

---

### P8 · Hallucination Detection & Mitigation

**Overview:** Complete detect-classify-mitigate pipeline with 3 detectors at different cost/accuracy tradeoffs and 3 mitigation strategies, evaluated on a 25-claim labeled benchmark.

**Key features:** Precision/recall/F1 + AUC/ROC curves per detector. Mitigation success rate per strategy. Gradio demo for live hallucination scanning.

**Highlights:** Three detectors in one pipeline — rule-based (free, instant), LLM judge (high recall), entailment/NLI (semantic). Each has different precision/recall characteristics visible in the ROC chart. The mitigation loop closes by re-scoring after each fix.

📁 [`projects/hallucination-detection/`](projects/hallucination-detection/)

---

### P9 · AI Document Intelligence System

**Overview:** End-to-end document processing pipeline: multi-format ingestion (TXT/PDF/DOCX/CSV), LLM classification, structured extraction, intelligent chunking, TF-IDF/dense indexing, RAG-based QA with citations, and accuracy benchmarking against ground truth.

**Key features:** Accuracy benchmark across 5 document types (contract, invoice, research report, meeting minutes, financial statement). Gradio demo with file upload. `ground_truth.json` with 25 QA pairs for reproducible evaluation.

**Highlight:** `ingestion.py` has layered fallbacks at every level (pypdf → pdfminer → warning). `chunker.py` preserves section heading context in chunk metadata. The pipeline runs on zero ML dependencies by default — sentence-transformers is a one-flag upgrade.

📁 [`projects/document-intelligence/`](projects/document-intelligence/)

---

## Reproducibility

All experiments are fully reproducible:

```bash
# Clone and install
git clone https://github.com/ChuksForge/prompt-engineering-lab
cd prompt-engineering-lab

# Install the prompt testing framework (used across projects)
cd projects/prompt-testing-framework
pip install -e .

# Set API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENROUTER_API_KEY="sk-or-..."

# Run any project
cd projects/summarization-benchmark
pip install -r requirements.txt
python run_experiment.py --quick --models openai
python visualize.py
python update_findings.py
```

Each project's `results/` directory is git-ignored. Run the experiment scripts to regenerate all results locally.

---

## Cross-Project Architecture

Several components are shared or referenced across projects:

| Component | Built in | Used by |
|-----------|----------|---------|
| ROUGE evaluation | P1 | P2, P6 |
| Programmatic constraint rubric | P3 | P4 (RubricScorer) |
| `promptlab` batch runner | P4 | Referenced by P7 |
| TF-IDF retriever | P5 | P8, P9 |
| Tone detection | P6 | — |
| Cost-per-quality metric | P7 | — |
| Hallucination taxonomy | P8 | P9 (post-extraction) |
| `update_findings.py` pattern | P1 | All subsequent projects |

---

## Contact

Built by ChuksForge as a prompt engineering portfolio project.
Each project README contains detailed methodology, results, and reproduction instructions.
