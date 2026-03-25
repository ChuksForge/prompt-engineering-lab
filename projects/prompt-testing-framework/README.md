# 🧪 P4 — Prompt Testing Framework (`promptlab`)

> **Installable Python library + CLI for evaluating, A/B comparing, and regression-testing LLM prompts**  
> Part of the [prompt-engineering-lab](../../README.md) portfolio

---

## Overview

`promptlab` is the infrastructure layer of this portfolio. It wraps OpenAI, Anthropic, and OpenRouter behind a single interface and provides three core capabilities every serious prompt engineer needs:

| Capability | What it does |
|------------|-------------|
| **Batch evaluation** | Test prompts across models and inputs with programmatic rubric scoring |
| **A/B comparison** | Compare two prompt variants with statistical significance testing |
| **Regression testing** | Save baseline snapshots, detect when prompt changes degrade performance |

---

## Install

```bash
# Editable install (recommended for development)
pip install -e .

# With YAML test file support
pip install -e ".[yaml]"
```

---

## Quick Start — Library API

```python
from promptlab import PromptLab

lab = PromptLab(models=["gpt-4o-mini", "claude-haiku-4-5-20251001"])

# 1. Batch evaluation
batch = lab.run(
    prompts={
        "v1": "Summarize: {{text}}",
        "v2": "Write a 2-sentence summary: {{text}}",
    },
    inputs=[{"id": "doc1", "text": "..."}],
    checks=[
        PromptLab.word_limit(80),
        PromptLab.no_refusal(),
        PromptLab.must_not_contain("As an AI"),
    ],
)
batch.report.print_summary()
batch.save("results/").plot()

# 2. A/B comparison
reports = lab.ab(
    prompt_a="Summarize: {{text}}",
    prompt_b="You are an expert editor. Write a 2-sentence summary: {{text}}",
    inputs=[{"id": "doc1", "text": "..."}],
    checks=[PromptLab.word_limit(60)],
)
for model, report in reports.items():
    print(report.summary())

# 3. Regression testing
from promptlab.scorers import RubricScorer

scorer = RubricScorer(checks=[PromptLab.word_limit(80), PromptLab.no_refusal()])

lab.regression.save_baseline(
    name="summarizer_v1",
    prompt="Summarize in 2 sentences: {{text}}",
    inputs=[...],
    models=["gpt-4o-mini"],
    scorer=scorer,
)

report = lab.regression.check(
    name="summarizer_v1",
    prompt="Give a brief 2-sentence summary: {{text}}",  # modified
    inputs=[...],
    models=["gpt-4o-mini"],
    scorer=scorer,
)
print(report.summary())
if report.has_regression:
    raise SystemExit(1)  # use in CI
```

---

## Quick Start — CLI

```bash
# Run a test suite from YAML
promptlab run tests/summarizer.yaml
promptlab run tests/summarizer.yaml --models gpt-4o-mini --output results/

# A/B compare inline
promptlab ab \
  --prompt-a "Summarize: {{text}}" \
  --prompt-b "Write a 2-sentence summary: {{text}}" \
  --text "Scientists found that gut bacteria reduces anxiety in mice..."

# Regression: save then check
promptlab regression save --name summarizer_v1 --test tests/summarizer.yaml
promptlab regression check --name summarizer_v1 --test tests/summarizer.yaml

# List saved baselines
promptlab list
```

---

## Project Structure

```
prompt-testing-framework/
├── promptlab/
│   ├── __init__.py     ← Public API
│   ├── lab.py          ← PromptLab facade (main entry point)
│   ├── client.py       ← Unified multi-provider API client
│   ├── runner.py       ← Batch evaluation engine + template rendering
│   ├── scorers.py      ← RubricScorer · LLMJudgeScorer · CompositeScorer
│   ├── ab.py           ← A/B comparison + statistical significance
│   ├── regression.py   ← Baseline snapshots + drift detection
│   └── report.py       ← CSV/JSON export + chart generation
├── cli.py              ← CLI entry point
├── setup.py            ← pip install -e .
├── tests/
│   ├── summarizer.yaml          ← Example test suite (P1 use case)
│   └── instruction_following.yaml  ← Example test suite (P3 use case)
├── baselines/          ← Committed regression snapshots (git-tracked)
├── results/            ← Output files (git-ignored)
└── experiment.ipynb    ← Full framework demo notebook
```

---

## Built-in Rubric Checks

```python
PromptLab.word_limit(100)              # output <= 100 words
PromptLab.word_minimum(20)             # output >= 20 words
PromptLab.must_contain("conclusion")   # phrase must appear
PromptLab.must_not_contain("I cannot") # phrase must not appear
PromptLab.no_refusal()                 # no AI refusal phrases
PromptLab.json_valid()                 # output is valid JSON
PromptLab.numbered_list(3)             # at least 3 numbered items
PromptLab.contains_pattern(r"\d+%")   # regex must match
```

Custom checks:
```python
from promptlab.scorers import RubricScorer, RubricCheck

scorer = RubricScorer(checks=[
    RubricCheck("has_numbers", lambda o, _: bool(re.search(r'\d+', o))),
    RubricCheck("short_enough", lambda o, _: len(o.split()) <= 50, weight=2.0),
])
```

---

## Supported Models

| Provider | Models |
|----------|--------|
| OpenAI | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo` |
| Anthropic | `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`, `claude-opus-4-6` |
| OpenRouter | Any model via `mistralai/...`, `meta-llama/...`, `google/...` prefix |

---

## YAML Test File Format

```yaml
prompts:
  v1: "Summarize: {{text}}"
  v2: "Write a 2-sentence summary: {{text}}"

models:
  - gpt-4o-mini
  - claude-haiku-4-5-20251001

inputs:
  - id: doc1
    text: "Your source text here..."

checks:
  - type: word_limit
    n: 80
  - type: must_not_contain
    phrase: "As an AI"
  - type: no_refusal

llm_judge: false
```

---

## Used by Other Projects

| Project | How it uses promptlab |
|---------|----------------------|
| P1 Summarization Benchmark | `BatchRunner` + `RubricScorer` for multi-model eval |
| P2 Style Transfer | `BatchRunner` for cross-model runs |
| P3 Instruction Following | `RubricScorer` constraint engine (shared logic) |
| P7 LLM Benchmark System | `ABComparison` for full leaderboard generation |
| P8 Hallucination Detection | `RegressionTracker` to monitor detection accuracy |

---

## Related Projects

- **P3:** [Instruction Following](../instruction-following-benchmark/) — constraint engine adapted from `RubricScorer`
- **P7:** [LLM Benchmark System](../prompt-benchmark-system/) — uses `ABComparison` at scale

---

*prompt-engineering-lab / projects / prompt-testing-framework*
