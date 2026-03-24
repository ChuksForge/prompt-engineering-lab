"""
detectors/entailment.py
=======================
Hallucination Detection — Entailment-Based Detector
Project: P8 · prompt-engineering-lab

Checks whether the source ENTAILS the claim using:
  Primary:  sentence-transformers NLI model (if available)
  Fallback: TF-IDF cosine similarity (zero-dependency)

NLI labels: ENTAILMENT (supports) / NEUTRAL / CONTRADICTION (refutes)
A claim is flagged as hallucination if:
  - Label is CONTRADICTION, or
  - Label is NEUTRAL and similarity is low (claim introduces new info)
"""

import re
import math
import logging
from dataclasses import dataclass, field
from typing import Optional

from .rule_based import DetectionResult

logger = logging.getLogger(__name__)

# Similarity threshold below which NEUTRAL → flagged
NEUTRAL_SIMILARITY_THRESHOLD = 0.35


# ── TF-IDF cosine fallback ───────────────────────────────────

def _tokenize(text: str) -> list:
    return re.findall(r"\b[a-z]+\b", text.lower())

def _tfidf_vector(text: str, vocab: list) -> list:
    tokens = _tokenize(text)
    tf = {t: tokens.count(t) / len(tokens) for t in set(tokens)} if tokens else {}
    return [tf.get(w, 0.0) for w in vocab]

def _cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x**2 for x in a))
    nb  = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb) if na * nb else 0.0

def _cosine_similarity(text_a: str, text_b: str) -> float:
    vocab = list(set(_tokenize(text_a + " " + text_b)))
    va = _tfidf_vector(text_a, vocab)
    vb = _tfidf_vector(text_b, vocab)
    return round(_cosine(va, vb), 4)


# ── NLI via sentence-transformers ───────────────────────────

def _nli_predict(source: str, claim: str):
    """
    Returns (label, score) using cross-encoder NLI model.
    Labels: ENTAILMENT, NEUTRAL, CONTRADICTION
    Requires: pip install sentence-transformers
    """
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    scores = model.predict([(source, claim)])
    label_map = {0: "CONTRADICTION", 1: "ENTAILMENT", 2: "NEUTRAL"}
    label_id = scores[0].argmax()
    label = label_map.get(int(label_id), "NEUTRAL")
    confidence = float(scores[0][label_id])
    return label, round(confidence, 3)


class EntailmentDetector:
    """
    Entailment-based hallucination detector.

    Uses NLI to check if source text entails (supports) the claim.
    Falls back to cosine similarity if sentence-transformers unavailable.

    Args:
        use_ml: If True, attempts to load NLI model. If False or unavailable,
                uses TF-IDF cosine similarity fallback.
        neutral_threshold: Similarity score below which NEUTRAL is flagged.
    """

    def __init__(
        self,
        use_ml: bool = True,
        neutral_threshold: float = NEUTRAL_SIMILARITY_THRESHOLD,
    ):
        self.use_ml            = use_ml
        self.neutral_threshold = neutral_threshold
        self._ml_available     = False

        if use_ml:
            try:
                import sentence_transformers  # noqa
                self._ml_available = True
                logger.info("EntailmentDetector: NLI model available")
            except ImportError:
                logger.info("EntailmentDetector: sentence-transformers not installed, using cosine fallback")

    def detect(
        self,
        claim: str,
        source: str,
        claim_id: str = "unknown",
        ground_truth: Optional[bool] = None,
    ) -> DetectionResult:

        if self._ml_available:
            label, conf = _nli_predict(source, claim)
            similarity  = _cosine_similarity(source, claim)

            if label == "CONTRADICTION":
                is_hallucination = True
                confidence = conf
                signals = [f"nli:CONTRADICTION:{conf:.2f}"]
                explanation = f"NLI model detected CONTRADICTION (score={conf:.2f})"

            elif label == "NEUTRAL" and similarity < self.neutral_threshold:
                is_hallucination = True
                confidence = round((1 - similarity) * 0.6, 3)
                signals = [f"nli:NEUTRAL", f"low_similarity:{similarity:.2f}"]
                explanation = f"NLI NEUTRAL + low similarity ({similarity:.2f}) — claim introduces unsupported content"

            else:
                is_hallucination = False
                confidence = conf if label == "ENTAILMENT" else 0.3
                signals = [f"nli:{label}:{conf:.2f}"]
                explanation = f"NLI: {label} (score={conf:.2f})"

        else:
            # Cosine similarity fallback
            similarity = _cosine_similarity(source, claim)

            if similarity < self.neutral_threshold:
                is_hallucination = True
                confidence = round(1 - similarity, 3)
                signals = [f"low_cosine_similarity:{similarity:.2f}"]
                explanation = f"Low semantic similarity to source ({similarity:.2f}) — claim may introduce unsupported content"
            else:
                is_hallucination = False
                confidence = round(similarity, 3)
                signals = [f"cosine_similarity:{similarity:.2f}"]
                explanation = f"Sufficient semantic similarity to source ({similarity:.2f})"

        return DetectionResult(
            claim_id=claim_id,
            detector="entailment" + ("_nli" if self._ml_available else "_cosine"),
            is_hallucination=is_hallucination,
            confidence=confidence,
            signals=signals,
            explanation=explanation,
            ground_truth=ground_truth,
        )

    def detect_batch(self, cases: list) -> list:
        return [
            self.detect(
                claim=c["claim"],
                source=c["source_context"],
                claim_id=c["claim_id"],
                ground_truth=bool(c.get("is_hallucination")),
            )
            for c in cases
        ]


if __name__ == "__main__":
    detector = EntailmentDetector(use_ml=False)  # cosine fallback

    tests = [
        ("CLEAN",    "The study involved 3,200 participants.", "3,200 participants in a 12-week study.", False),
        ("WRONG_NUM","The study involved 5,000 participants.", "3,200 participants in a 12-week study.", True),
        ("UNRELATED","The moon is made of cheese.",            "3,200 participants in a 12-week study.", True),
    ]
    for name, claim, source, gt in tests:
        r = detector.detect(claim, source, ground_truth=gt)
        icon = "✓" if r.correct else "✗"
        print(f"{icon} {name:12s}  hal={r.is_hallucination}  conf={r.confidence:.2f}  {r.explanation[:60]}")
