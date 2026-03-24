"""
pipeline.py
===========
Hallucination Detection & Mitigation — Main Pipeline
Project: P8 · prompt-engineering-lab

Orchestrates:
  1. Load benchmark dataset
  2. Run all detectors on all claims
  3. Compute precision/recall/F1/AUC per detector
  4. Run mitigation on detected hallucinations
  5. Re-score after mitigation
  6. Save results

Usage:
    python pipeline.py                        # full run
    python pipeline.py --detectors rule_based # one detector
    python pipeline.py --quick               # 10 claims only
    python pipeline.py --no-mitigate         # skip mitigation
    python pipeline.py --models openai       # API provider filter
"""

import os
import time
import logging
import argparse
from pathlib import Path

import pandas as pd

from detectors import RuleBasedDetector, LLMJudgeDetector, EntailmentDetector
from evaluation import (
    compute_metrics, compute_roc_data,
    compute_type_breakdown, compute_mitigation_summary
)
from mitigator import Mitigator

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
        "claude-haiku-4-5-20251001": {"provider": "anthropic", "label": "Claude Haiku"},
    },
    "openrouter": {
        "google/gemini-2.0-flash-001":  {"provider": "openrouter", "label": "Gemini 2.0 Flash"},
    },
}


# ── Client init ──────────────────────────────────────────────

def init_clients(model_filter):
    clients, active = {}, set()
    needed = set()
    for grp, grp_models in MODELS.items():
        if model_filter and grp not in model_filter:
            continue
        for _, meta in grp_models.items():
            needed.add(meta["provider"])

    from openai import OpenAI
    import anthropic as ant

    if "openai" in needed and os.environ.get("OPENAI_API_KEY"):
        try:
            clients["openai"] = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            active.add("openai"); logger.info("  openai ready")
        except Exception as e:
            logger.warning(f"  openai: {e}")

    if "anthropic" in needed and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            clients["anthropic"] = ant.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            active.add("anthropic"); logger.info("  anthropic ready")
        except Exception as e:
            logger.warning(f"  anthropic: {e}")

    if "openrouter" in needed and os.environ.get("OPENROUTER_API_KEY"):
        try:
            clients["openrouter"] = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
            active.add("openrouter"); logger.info("  openrouter ready")
        except Exception as e:
            logger.warning(f"  openrouter: {e}")

    return clients, active


# ── Main pipeline ────────────────────────────────────────────

def run_pipeline(
    model_filter=None,
    detector_filter=None,
    quick=False,
    run_mitigation=True,
    use_ml_entailment=False,
):
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load benchmark
    df = pd.read_csv(DATA_DIR / "benchmark.csv")
    if quick:
        df = df.head(10)
        logger.info("Quick mode: 10 claims")

    cases = df.to_dict("records")
    type_map = {c["claim_id"]: c.get("hallucination_type","none") for c in cases}

    # Init clients
    clients, active = init_clients(model_filter)

    # ── Detectors ────────────────────────────────────────────
    detectors = {}

    if not detector_filter or "rule_based" in detector_filter:
        detectors["rule_based"] = RuleBasedDetector()

    if not detector_filter or "entailment" in detector_filter:
        detectors["entailment"] = EntailmentDetector(use_ml=use_ml_entailment)

    if (not detector_filter or "llm_judge" in detector_filter) and active:
        # Use first available provider
        for grp, grp_models in MODELS.items():
            if model_filter and grp not in model_filter:
                continue
            for model_id, mmeta in grp_models.items():
                provider = mmeta["provider"]
                if provider not in active:
                    continue
                client = clients[provider]
                # Adapt client for LLMJudgeDetector
                if provider == "anthropic":
                    judge_client = client
                else:
                    judge_client = client
                detectors["llm_judge"] = LLMJudgeDetector(
                    client=judge_client,
                    judge_model=model_id,
                    provider=provider,
                )
                logger.info(f"  LLM judge: {mmeta['label']} ({provider})")
                break
            if "llm_judge" in detectors:
                break

    logger.info(f"\n  Detectors: {list(detectors.keys())}")
    logger.info(f"  Claims:    {len(cases)}\n")

    # ── Run detection ────────────────────────────────────────
    all_detection_results = {}
    all_metrics           = {}
    all_roc               = {}
    all_type_breakdown    = {}

    for det_name, detector in detectors.items():
        logger.info(f"  Running {det_name}...")
        results = detector.detect_batch(cases)
        all_detection_results[det_name] = results

        metrics        = compute_metrics(results, det_name)
        roc            = compute_roc_data(results)
        type_breakdown = compute_type_breakdown(results, type_map)

        all_metrics[det_name]        = metrics
        all_roc[det_name]            = roc
        all_type_breakdown[det_name] = type_breakdown

        logger.info(
            f"    P={metrics.precision:.3f}  R={metrics.recall:.3f}  "
            f"F1={metrics.f1:.3f}  AUC={roc['auc']:.3f}"
        )

    # Save detection results
    det_rows = []
    for det_name, results in all_detection_results.items():
        for r in results:
            row = {
                "detector":         det_name,
                "claim_id":         r.claim_id,
                "predicted_hal":    r.is_hallucination,
                "ground_truth_hal": r.ground_truth,
                "confidence":       r.confidence,
                "correct":          r.correct,
                "explanation":      r.explanation,
                "signals":          str(r.signals),
            }
            det_rows.append(row)
    pd.DataFrame(det_rows).to_csv(RESULTS_DIR / "detection_results.csv", index=False)

    # Save metrics
    metrics_rows = [m.to_dict() for m in all_metrics.values()]
    pd.DataFrame(metrics_rows).to_csv(RESULTS_DIR / "detector_metrics.csv", index=False)
    logger.info("\n  detection_results.csv saved")
    logger.info("  detector_metrics.csv saved")

    # Save ROC data
    roc_rows = []
    for det_name, roc in all_roc.items():
        for i in range(len(roc["thresholds"])):
            roc_rows.append({
                "detector":  det_name,
                "threshold": roc["thresholds"][i],
                "tpr":       roc["tpr"][i],
                "fpr":       roc["fpr"][i],
                "auc":       roc["auc"],
            })
    pd.DataFrame(roc_rows).to_csv(RESULTS_DIR / "roc_data.csv", index=False)
    logger.info("  roc_data.csv saved")

    # ── Mitigation ───────────────────────────────────────────
    if run_mitigation and active:
        logger.info("\n  Running mitigation pipeline...")

        # Use rule_based for re-scoring (zero cost)
        rescore_detector = detectors.get("rule_based", RuleBasedDetector())

        # Pick mitigation model
        mit_client = mit_provider = mit_model = None
        for grp, grp_models in MODELS.items():
            if model_filter and grp not in model_filter:
                continue
            for model_id, mmeta in grp_models.items():
                if mmeta["provider"] in active:
                    mit_provider = mmeta["provider"]
                    mit_model    = model_id
                    mit_client   = clients[mit_provider]
                    break
            if mit_client:
                break

        if mit_client:
            mitigator = Mitigator(
                client=mit_client,
                model=mit_model,
                provider=mit_provider,
                detector=rescore_detector,
            )

            # Find detected hallucinations from best detector
            best_det = max(all_metrics, key=lambda d: all_metrics[d].f1)
            detected_hals = [
                r for r in all_detection_results[best_det]
                if r.is_hallucination
            ][:8]  # limit to 8 for cost control

            logger.info(f"  Mitigating {len(detected_hals)} detected hallucinations using {best_det}")

            mit_results = []
            for r in detected_hals:
                case = next((c for c in cases if c["claim_id"] == r.claim_id), None)
                if not case:
                    continue
                logger.info(f"    [{r.claim_id}] all strategies...")
                results = mitigator.mitigate_all_strategies(
                    claim=case["claim"],
                    source=case["source_context"],
                    original_confidence=r.confidence,
                    claim_id=r.claim_id,
                )
                mit_results.extend(results)
                time.sleep(0.5)

            mit_rows = [r.to_dict() for r in mit_results]
            pd.DataFrame(mit_rows).to_csv(RESULTS_DIR / "mitigation_results.csv", index=False)

            mit_summary = compute_mitigation_summary(mit_results)
            pd.DataFrame([
                {"strategy": s, **v}
                for s, v in mit_summary.items()
            ]).to_csv(RESULTS_DIR / "mitigation_summary.csv", index=False)

            logger.info("  mitigation_results.csv saved")
            logger.info("  mitigation_summary.csv saved")

    logger.info("\n  Pipeline complete.")
    return all_metrics, all_roc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",     type=str)
    parser.add_argument("--detectors",  type=str)
    parser.add_argument("--quick",      action="store_true")
    parser.add_argument("--no-mitigate", action="store_true")
    parser.add_argument("--use-ml",     action="store_true", help="Use sentence-transformers for entailment")
    args = parser.parse_args()

    run_pipeline(
        model_filter     = args.models.split(",")    if args.models    else None,
        detector_filter  = args.detectors.split(",") if args.detectors else None,
        quick            = args.quick,
        run_mitigation   = not args.no_mitigate,
        use_ml_entailment= args.use_ml,
    )
