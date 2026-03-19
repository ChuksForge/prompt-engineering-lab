# 📋 P3 — Instruction Following Benchmark

> **Programmatic rubric-based evaluation of instruction compliance across three failure-prone categories**  
> Part of the [prompt-engineering-lab](../../../../README.md) portfolio

---

## Overview

Models often claim to follow instructions while silently violating them. This benchmark makes compliance **machine-checkable** — no subjective LLM judge needed for scoring. Every pass/fail decision traces to a specific constraint with a reason.

| | |
|---|---|
| **Categories** | Multi-Step Instructions · Tone & Persona · Negation Handling |
| **Tasks** | 18 tasks — 6 per category, graded easy/medium/hard |
| **Models** | GPT-4o-mini · GPT-4o · Claude Haiku · Claude Sonnet 4.6 · Mistral 7B · Llama 3 8B |
| **Scoring** | Programmatic constraint rubric (16 constraint types) |
| **Key Output** | Pass rate · Full compliance % · Failure taxonomy |

---

## Results

![Instruction Following Benchmark Results](results/charts.png)

### Leaderboard

| Rank | Model | Overall | Multi-Step | Tone & Persona | Negation | Full Compliance |
|------|-------|---------|------------|----------------|----------|-----------------|
| 1 | GPT-4o-mini | 96.7% | 93.3% | 96.7% | 100.0% | 83.3% |
| 2 | Claude Sonnet 4.6 | 95.3% | 89.2% | 96.7% | 100.0% | 77.8% |
| 3 | Claude Haiku | 94.8% | 93.3% | 96.7% | 94.4% | 83.3% |
| 4 | GPT-4o | 94.2% | 90.0% | 96.7% | 95.8% | 77.8% |
| 5 | Llama 3 8B | 92.8% | 90.0% | 96.7% | 91.7% | 72.2% |
| 6 | Mistral small creative | 88.8% | 93.3% | 91.1% | 81.9% | 66.7% |

*Run `python update_findings.py` after the experiment to populate.*

### Failure Mode Analysis

| Model | Top Failure Mode | 2nd Mode | 3rd Mode |
|-------|-----------------|----------|----------|
| GPT-4o-mini | MISSED_STEP (33%) | WRONG_FORMAT (33%) | LENGTH_VIOLATION (33%) |
| GPT-4o | MISSED_STEP (40%) | LENGTH_VIOLATION (40%) | WRONG_FORMAT (20%) |
| Claude Haiku | LENGTH_VIOLATION (50%) | WRONG_FORMAT (25%) | MISSED_STEP (25%) |
| Claude Sonnet 4.6 | WRONG_FORMAT (75%) | LENGTH_VIOLATION (25%) | — |
| Mistral small creative | LENGTH_VIOLATION (43%) | WRONG_FORMAT (14%) | MISSED_STEP (14%) |
| Llama 3 8B | MISSED_STEP (33%) | LENGTH_VIOLATION (33%) | WRONG_FORMAT (17%) |

---

## Project Structure

```
instruction-following/
├── experiment.ipynb       ← Main analysis notebook
├── run_experiment.py      ← CLI runner
├── evaluation.py          ← Constraint checker + failure taxonomy (16 constraint types)
├── visualize.py           ← 6 charts + hero image
├── update_findings.py     ← Auto-populate README + print notebook findings
├── data/
│   └── tasks.csv          ← 18 tasks with machine-checkable constraint rubrics
└── results/
    ├── results.csv
    ├── leaderboard.csv
    ├── failure_report.csv
    ├── charts.png
    ├── chart_leaderboard.png
    ├── chart_by_category.png
    ├── chart_by_difficulty.png
    ├── chart_failure_modes.png
    └── chart_task_heatmap.png
```

---

## Quick Start

```bash
pip install -r requirements.txt

export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENROUTER_API_KEY="sk-or-..."

# Quick sanity check
python run_experiment.py --quick

# Full run
python run_experiment.py

# Charts
python visualize.py

# Auto-fill README + findings
python update_findings.py

# Explore in Notebook
jupyter notebook experiment.ipynb
```

---

## CLI Options

```
python run_experiment.py [options]

  --models      openai,anthropic,openrouter
  --categories  multi_step,tone_persona,negation
  --difficulty  easy,medium,hard
  --tasks       MS01,TP03,NG06
  --quick       6 tasks, openai only
```

---

## Constraint Types

The evaluation engine supports 16 programmatic constraint types:

| Type | Description | Failure Mode |
|------|-------------|-------------|
| `step_present` | Keyword/regex must appear in output | MISSED_STEP |
| `exact_phrase` | Verbatim string must appear | MISSED_STEP |
| `word_absent` | Word(s) must NOT appear | VIOLATED_NEGATION |
| `char_absent` | Character must NOT appear | VIOLATED_NEGATION |
| `tone_word_present` | At least one tone word must appear | WRONG_TONE |
| `tone_word_absent` | None of the tone words may appear | WRONG_TONE |
| `numbered_list` | Must contain N numbered items | WRONG_FORMAT |
| `paragraph_count` | Must have exactly N paragraphs | WRONG_FORMAT |
| `word_count_min` | Must have >= N words | LENGTH_VIOLATION |
| `word_count_max` | Must have <= N words | LENGTH_VIOLATION |
| `step_count` | Keyword must appear >= N times | MISSED_STEP |
| `allocation_sum` | Percentages must sum to target | WRONG_FORMAT |
| `contains_pattern` | Regex pattern must match | MISSED_STEP |
| `starts_with_caps_headline` | First line must be ALL CAPS | WRONG_FORMAT |
| `sentence_not_starts_with` | No sentence starts with given word | VIOLATED_NEGATION |
| `not_starts_with_question` | Output must not open with a question | VIOLATED_NEGATION |

---

## Failure Taxonomy

| Mode | Meaning |
|------|---------|
| `FULL_COMPLIANCE` | All constraints passed |
| `MISSED_STEP` | Required content/step was absent |
| `VIOLATED_NEGATION` | Used a forbidden word, phrase, or character |
| `WRONG_FORMAT` | Wrong structure — count, paragraphs, list format |
| `WRONG_TONE` | Tone or persona not maintained |
| `LENGTH_VIOLATION` | Output too long or too short |
| `PARTIAL_COMPLIANCE` | Multiple failure modes present |

---

## Related Projects

- **P1:** [Summarization Benchmark](../../../summarization-benchmark) — shares eval infrastructure patterns
- **P6:** [Prompt Testing Framework](../../../prompt-testing-framework) — wraps this constraint engine into a reusable library

---

*prompt-engineering-lab / projects / instruction-following*
