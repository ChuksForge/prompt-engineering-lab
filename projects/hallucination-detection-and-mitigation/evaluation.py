"""
evaluation.py
=============
Hallucination Detection & Mitigation — Evaluation Engine
Project: P8 · prompt-engineering-lab

Computes:
  - Precision, Recall, F1 per detector
  - ROC curve data (TPR vs FPR at varying thresholds)
  - Per hallucination-type breakdown
  - Mitigation success rate per strategy
"""

import math
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class DetectorMetrics:
    detector: str
    tp: int = 0    # correctly flagged hallucinations
    fp: int = 0    # incorrectly flagged clean claims
    tn: int = 0    # correctly cleared clean claims
    fn: int = 0    # missed hallucinations

    @property
    def precision(self) -> float:
        return round(self.tp / (self.tp + self.fp), 4) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return round(self.tp / (self.tp + self.fn), 4) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return round(2 * p * r / (p + r), 4) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return round((self.tp + self.tn) / total, 4) if total else 0.0

    @property
    def false_positive_rate(self) -> float:
        return round(self.fp / (self.fp + self.tn), 4) if (self.fp + self.tn) else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update({
            "precision": self.precision,
            "recall":    self.recall,
            "f1":        self.f1,
            "accuracy":  self.accuracy,
            "fpr":       self.false_positive_rate,
        })
        return d


def compute_metrics(results: list, detector_name: str) -> DetectorMetrics:
    """
    Compute confusion matrix metrics from a list of DetectionResult objects.
    Only processes results where ground_truth is not None.
    """
    m = DetectorMetrics(detector=detector_name)
    for r in results:
        if r.ground_truth is None:
            continue
        predicted = r.is_hallucination
        actual    = r.ground_truth
        if predicted and actual:
            m.tp += 1
        elif predicted and not actual:
            m.fp += 1
        elif not predicted and not actual:
            m.tn += 1
        else:
            m.fn += 1
    return m


def compute_roc_data(results: list) -> dict:
    """
    Compute ROC curve data for a detector by varying confidence threshold.
    Returns {"thresholds": [...], "tpr": [...], "fpr": [...], "auc": float}
    """
    import numpy as np

    # Filter to labeled results
    labeled = [(r.confidence, r.is_hallucination, r.ground_truth)
               for r in results if r.ground_truth is not None]
    if not labeled:
        return {"thresholds": [], "tpr": [], "fpr": [], "auc": 0.0}

    confidences, predictions, ground_truths = zip(*labeled)
    thresholds = sorted(set(confidences), reverse=True)
    thresholds = [0.0] + list(thresholds) + [1.0]

    tpr_list, fpr_list = [], []
    n_pos = sum(gt for gt in ground_truths)
    n_neg = len(ground_truths) - n_pos

    for thresh in thresholds:
        # At this threshold, flag as hallucination if confidence >= thresh
        tp = sum(1 for c, p, gt in labeled if c >= thresh and gt)
        fp = sum(1 for c, p, gt in labeled if c >= thresh and not gt)
        tpr = tp / n_pos if n_pos else 0.0
        fpr = fp / n_neg if n_neg else 0.0
        tpr_list.append(round(tpr, 4))
        fpr_list.append(round(fpr, 4))

    # AUC via trapezoidal rule
    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += abs(fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2
    auc = round(auc, 4)

    return {
        "thresholds": [round(t, 3) for t in thresholds],
        "tpr":        tpr_list,
        "fpr":        fpr_list,
        "auc":        auc,
    }


def compute_type_breakdown(results: list, ground_truth_types: dict) -> dict:
    """
    Breakdown of detection rate per hallucination type.
    ground_truth_types: { claim_id: hallucination_type }
    Returns: { type: { detected: int, missed: int, detection_rate: float } }
    """
    breakdown = {}
    for r in results:
        if r.ground_truth is None or not r.ground_truth:
            continue  # only care about actual hallucinations
        htype = ground_truth_types.get(r.claim_id, "unknown")
        if htype not in breakdown:
            breakdown[htype] = {"detected": 0, "missed": 0}
        if r.is_hallucination:
            breakdown[htype]["detected"] += 1
        else:
            breakdown[htype]["missed"] += 1

    for htype in breakdown:
        total = breakdown[htype]["detected"] + breakdown[htype]["missed"]
        breakdown[htype]["detection_rate"] = round(
            breakdown[htype]["detected"] / total, 3
        ) if total else 0.0

    return breakdown


def compute_mitigation_summary(mitigation_results: list) -> dict:
    """
    Summarize mitigation effectiveness per strategy.
    Returns: { strategy: { success_rate, avg_improvement, still_hallucinating_rate } }
    """
    by_strategy = {}
    for mr in mitigation_results:
        s = mr.strategy if hasattr(mr, "strategy") else mr.get("strategy", "unknown")
        if s not in by_strategy:
            by_strategy[s] = []
        by_strategy[s].append(mr)

    summary = {}
    for strategy, results in by_strategy.items():
        total = len(results)
        still_hal = sum(
            1 for r in results
            if (r.still_hallucinating if hasattr(r, "still_hallucinating") else r.get("still_hallucinating", True))
        )
        improvements = [
            (r.improvement if hasattr(r, "improvement") else r.get("improvement", 0))
            for r in results
        ]
        summary[strategy] = {
            "total":               total,
            "success_rate":        round(1 - still_hal/total, 3) if total else 0.0,
            "still_hallucinating": still_hal,
            "avg_improvement":     round(sum(improvements)/total, 4) if total else 0.0,
        }
    return summary
