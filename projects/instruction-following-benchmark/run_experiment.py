"""
run_experiment.py
=================
Instruction Following Benchmark — Experiment Runner
Project: P3 · prompt-engineering-lab

Usage:
    python run_experiment.py                            # full run
    python run_experiment.py --models openai            # one provider
    python run_experiment.py --categories multi_step    # one category
    python run_experiment.py --difficulty easy          # filter difficulty
    python run_experiment.py --quick                    # 6 tasks, openai only
    python run_experiment.py --tasks MS01,TP01,NG01     # specific tasks

Outputs:
    results/results.csv       — full results, one row per model×task
    results/leaderboard.csv   — aggregated pass rates ranked
    results/failure_report.csv — failure mode breakdown
"""

import os
import re
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

from evaluation import evaluate_output

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
DATA_DIR    = Path("data")

MODELS = {
    "openai": {
        "gpt-4o-mini": {"provider": "openai",     "label": "GPT-4o-mini"},
        "gpt-4o":      {"provider": "openai",     "label": "GPT-4o"},
    },
    "anthropic": {
        "claude-haiku-4-5-20251001":    {"provider": "anthropic", "label": "Claude Haiku"},
        "claude-sonnet-4-6": {"provider": "anthropic", "label": "Claude Sonnet 4.6"},
    },
    "openrouter": {
        "mistralai/mistral-small-creative":  {"provider": "openrouter", "label": "Mistral small creative"},
        "meta-llama/llama-3-8b-instruct": {"provider": "openrouter", "label": "Llama 3 8B"},
    },
}


# ── Clients ──────────────────────────────────────────────────

def get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def get_anthropic_client():
    import anthropic
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def get_openrouter_client():
    from openai import OpenAI
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def call_model(provider: str, model_id: str, prompt: str, context: str, clients: dict) -> tuple:
    """Returns (output_text, latency_s)."""
    messages = []
    if context:
        messages.append({"role": "user", "content": f"Context:\n{context}\n\n{prompt}"})
    else:
        messages.append({"role": "user", "content": prompt})

    for attempt in range(3):
        try:
            t0 = time.time()

            if provider == "openai":
                resp = clients["openai"].chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=0.2,   # low temp for instruction following
                    max_tokens=600,
                )
                text = resp.choices[0].message.content.strip()

            elif provider == "anthropic":
                resp = clients["anthropic"].messages.create(
                    model=model_id,
                    max_tokens=600,
                    temperature=0.2,
                    messages=messages,
                )
                text = resp.content[0].text.strip()

            elif provider == "openrouter":
                resp = clients["openrouter"].chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=600,
                )
                text = resp.choices[0].message.content.strip()

            else:
                raise ValueError(f"Unknown provider: {provider}")

            return text, time.time() - t0

        except Exception as e:
            if attempt == 2:
                raise
            logger.warning(f"Attempt {attempt+1} failed: {e} — retrying in 2s")
            time.sleep(2)


# ── Leaderboard & failure report ─────────────────────────────

def build_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df[df["error"].isna() | (df["error"] == "")]

    lb = (
        df_clean
        .groupby(["model", "category"])
        .agg(
            avg_pass_rate=("pass_rate", "mean"),
            tasks_run=("task_id", "count"),
            full_compliance=("failure_taxonomy", lambda x: (x == "FULL_COMPLIANCE").sum()),
        )
        .round(4)
        .reset_index()
    )
    lb["compliance_rate"] = (lb["full_compliance"] / lb["tasks_run"]).round(4)

    # Overall across categories
    overall = (
        df_clean
        .groupby("model")
        .agg(
            avg_pass_rate=("pass_rate", "mean"),
            tasks_run=("task_id", "count"),
            full_compliance=("failure_taxonomy", lambda x: (x == "FULL_COMPLIANCE").sum()),
        )
        .round(4)
        .reset_index()
    )
    overall["category"] = "OVERALL"
    overall["compliance_rate"] = (overall["full_compliance"] / overall["tasks_run"]).round(4)

    combined = pd.concat([lb, overall], ignore_index=True)
    combined = combined.sort_values(["category", "avg_pass_rate"], ascending=[True, False])
    combined.insert(0, "rank", range(1, len(combined) + 1))
    return combined


def build_failure_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model breakdown of failure modes."""
    df_clean = df[df["error"].isna() | (df["error"] == "")]

    rows = []
    for model in df_clean["model"].unique():
        mdf = df_clean[df_clean["model"] == model]
        all_modes = []
        for modes_json in mdf["failure_modes"].dropna():
            try:
                modes = json.loads(modes_json)
                all_modes.extend(modes)
            except Exception:
                pass

        from collections import Counter
        mode_counts = Counter(all_modes)
        total_failures = sum(mode_counts.values())

        for mode, count in mode_counts.most_common():
            rows.append({
                "model": model,
                "failure_mode": mode,
                "count": count,
                "pct_of_failures": round(count / total_failures * 100, 1) if total_failures else 0,
            })

    return pd.DataFrame(rows)


# ── Main experiment loop ─────────────────────────────────────

def run_experiment(
    model_filter=None,
    category_filter=None,
    difficulty_filter=None,
    task_filter=None,
    quick=False,
) -> pd.DataFrame:

    RESULTS_DIR.mkdir(exist_ok=True)

    tasks = pd.read_csv(DATA_DIR / "tasks.csv")

    if quick:
        # One task per category per difficulty level
        tasks = tasks.groupby(["category","difficulty"]).first().reset_index().head(6)
        if not model_filter:
            model_filter = ["openai"]
        logger.info("Quick mode: 6 tasks, openai only")

    if task_filter:
        tasks = tasks[tasks["task_id"].isin(task_filter)]
    if category_filter:
        tasks = tasks[tasks["category"].isin(category_filter)]
    if difficulty_filter:
        tasks = tasks[tasks["difficulty"].isin(difficulty_filter)]

    # Init clients
    clients = {}
    active_providers = set()

    needed = set()
    for group, group_models in MODELS.items():
        if model_filter and group not in model_filter:
            continue
        for _, meta in group_models.items():
            needed.add(meta["provider"])

    init_map = {
        "openai":     ("OPENAI_API_KEY",     get_openai_client),
        "anthropic":  ("ANTHROPIC_API_KEY",  get_anthropic_client),
        "openrouter": ("OPENROUTER_API_KEY", get_openrouter_client),
    }
    for provider, (env_key, factory) in init_map.items():
        if provider in needed and os.environ.get(env_key):
            try:
                clients[provider] = factory()
                active_providers.add(provider)
                logger.info(f"  Client ready: {provider}")
            except Exception as e:
                logger.warning(f"  {provider} init failed: {e}")

    if not clients:
        raise RuntimeError("No API clients initialized. Check your environment variables.")

    all_results = []
    total = 0
    errors = 0

    for _, task in tasks.iterrows():
        for group, group_models in MODELS.items():
            if model_filter and group not in model_filter:
                continue
            for model_id, mmeta in group_models.items():
                provider = mmeta["provider"]
                if provider not in active_providers:
                    continue

                logger.info(
                    f"  [{task['task_id']}] {task['category']:12s} "
                    f"{task['difficulty']:6s} → {mmeta['label']}"
                )

                try:
                    output, latency = call_model(
                        provider, model_id,
                        task["instruction"],
                        str(task.get("context", "") or ""),
                        clients,
                    )

                    result = evaluate_output(
                        task_id=task["task_id"],
                        category=task["category"],
                        difficulty=task["difficulty"],
                        model=mmeta["label"],
                        output=output,
                        latency_s=latency,
                        constraints_json=task["constraints_json"],
                    )
                    total += 1

                except Exception as e:
                    logger.error(f"    FAILED: {e}")
                    from evaluation import EvalResult
                    result = EvalResult(
                        task_id=task["task_id"],
                        category=task["category"],
                        difficulty=task["difficulty"],
                        model=mmeta["label"],
                        output="",
                        latency_s=0.0,
                        error=str(e),
                    )
                    errors += 1

                all_results.append(result.to_dict())
                time.sleep(0.3)

    if not all_results:
        raise RuntimeError("No results generated.")

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "results.csv", index=False)
    logger.info(f"\n  results.csv saved  ({total} runs, {errors} errors)")

    lb = build_leaderboard(df)
    lb.to_csv(RESULTS_DIR / "leaderboard.csv", index=False)
    logger.info(f"  leaderboard.csv saved")

    fr = build_failure_report(df)
    fr.to_csv(RESULTS_DIR / "failure_report.csv", index=False)
    logger.info(f"  failure_report.csv saved")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",     type=str)
    parser.add_argument("--categories", type=str)
    parser.add_argument("--difficulty", type=str)
    parser.add_argument("--tasks",      type=str)
    parser.add_argument("--quick",      action="store_true")
    args = parser.parse_args()

    run_experiment(
        model_filter     = args.models.split(",")     if args.models     else None,
        category_filter  = args.categories.split(",") if args.categories else None,
        difficulty_filter= args.difficulty.split(",") if args.difficulty else None,
        task_filter      = args.tasks.split(",")      if args.tasks      else None,
        quick            = args.quick,
    )
