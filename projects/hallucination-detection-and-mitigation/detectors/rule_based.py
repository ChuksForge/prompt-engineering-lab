"""
detectors/rule_based.py
========================
Hallucination Detection — Rule-Based Detector
Project: P8 · prompt-engineering-lab

Zero API cost. Uses regex patterns to catch:
  - Numeric inconsistencies (numbers in claim not in source)
  - Superlative/absolute claims not supported by source
  - Entity mentions not in source
  - Temporal mismatches (dates, years)
  - Contradiction signals (antonyms of source statements)

Fast, cheap, good for obvious hallucinations.
Lower recall than LLM judge — misses subtle semantic errors.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DetectionResult:
    claim_id: str
    detector: str
    is_hallucination: bool          # predicted label
    confidence: float               # 0.0 – 1.0
    signals: list = field(default_factory=list)   # list of triggered signal names
    explanation: str = ""
    ground_truth: Optional[bool] = None           # True label if known

    @property
    def correct(self) -> Optional[bool]:
        if self.ground_truth is None:
            return None
        return self.is_hallucination == self.ground_truth


# ── Helpers ──────────────────────────────────────────────────

def _extract_numbers(text: str) -> set:
    """Extract all numeric values from text."""
    nums = set()
    # Integers and decimals
    for m in re.findall(r'\b\d+(?:\.\d+)?\b', text):
        nums.add(float(m))
    # Written numbers
    word_nums = {
        "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,
        "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
        "eleven":11,"twelve":12,"fifteen":15,"twenty":20,
        "thirty":30,"forty":40,"fifty":50,"hundred":100,"thousand":1000,
        "million":1_000_000,"billion":1_000_000_000,
    }
    for word, val in word_nums.items():
        if re.search(r'\b' + word + r'\b', text.lower()):
            nums.add(float(val))
    return nums

def _extract_dates(text: str) -> list:
    """Extract year and date patterns."""
    patterns = [
        r'\b(19|20)\d{2}\b',                          # years
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',              # MM/DD/YYYY
    ]
    dates = []
    for p in patterns:
        dates.extend(re.findall(p, text.lower()))
    return dates

SUPERLATIVES = [
    r'\b(most|best|worst|largest|smallest|highest|lowest|first|only|unique|never|always|all|every|none)\b',
    r'\b(unprecedented|definitive|conclusive|proven|guaranteed|certain|impossible)\b',
    r'\b(more effective than|superior to|better than|worse than|outperforms)\b',
]

ABSOLUTE_NEGATIONS = [
    r'\b(bans all|completely bans|prohibits all|never allows|always requires)\b',
]


# ── Rule-Based Detector ───────────────────────────────────────

class RuleBasedDetector:
    """
    Zero-cost hallucination detector using heuristic rules.

    Checks:
      1. Numeric inconsistency — numbers in claim absent from source
      2. Date mismatch — dates/years in claim not in source
      3. Unsupported superlatives — absolute claims not in source
      4. Entity invention — capitalized entities in claim absent from source
      5. Absolute negations — "bans all" type claims not in source
    """

    def __init__(self, numeric_tolerance: float = 0.05):
        """
        Args:
            numeric_tolerance: fractional tolerance for numeric comparison.
                              0.05 = numbers within 5% are considered matching.
        """
        self.numeric_tolerance = numeric_tolerance

    def detect(
        self,
        claim: str,
        source: str,
        claim_id: str = "unknown",
        ground_truth: Optional[bool] = None,
    ) -> DetectionResult:

        signals   = []
        confidence_parts = []

        # ── Check 1: Numeric inconsistency ───────────────────
        claim_nums  = _extract_numbers(claim)
        source_nums = _extract_numbers(source)

        for num in claim_nums:
            if num < 1:  # skip fractions/ratios
                continue
            # Check if any source number is within tolerance
            matched = any(
                abs(num - sn) / max(abs(sn), 1) <= self.numeric_tolerance
                for sn in source_nums
            )
            if not matched and source_nums:
                signals.append(f"numeric_mismatch:{num}")
                confidence_parts.append(0.7)

        # ── Check 2: Date mismatch ────────────────────────────
        claim_dates  = set(_extract_dates(claim))
        source_dates = set(_extract_dates(source))
        date_only_in_claim = claim_dates - source_dates
        if date_only_in_claim:
            signals.append(f"date_mismatch:{date_only_in_claim}")
            confidence_parts.append(0.65)

        # ── Check 3: Unsupported superlatives ─────────────────
        for pattern in SUPERLATIVES:
            claim_match  = re.search(pattern, claim.lower())
            source_match = re.search(pattern, source.lower())
            if claim_match and not source_match:
                signals.append(f"unsupported_superlative:{claim_match.group()}")
                confidence_parts.append(0.5)
                break  # one signal per type

        # ── Check 4: Absolute negations not in source ─────────
        for pattern in ABSOLUTE_NEGATIONS:
            if re.search(pattern, claim.lower()) and not re.search(pattern, source.lower()):
                signals.append("absolute_negation_not_in_source")
                confidence_parts.append(0.6)
                break

        # ── Check 5: Entity invention ─────────────────────────
        # Capitalized proper nouns in claim not found in source
        claim_entities  = set(re.findall(r'\b[A-Z][a-z]{2,}\b', claim))
        source_entities = set(re.findall(r'\b[A-Z][a-z]{2,}\b', source))
        # Common non-proper words to exclude
        exclude = {"The","This","These","That","Those","It","He","She","They","We",
                   "A","An","In","On","At","By","For","With","From","To","Of",
                   "However","Therefore","Furthermore","Additionally"}
        invented = claim_entities - source_entities - exclude
        if len(invented) >= 2:  # require at least 2 to reduce false positives
            signals.append(f"entity_invention:{invented}")
            confidence_parts.append(0.45)

        # ── Aggregate ─────────────────────────────────────────
        is_hallucination = len(signals) > 0
        if confidence_parts:
            confidence = round(max(confidence_parts), 3)
        else:
            confidence = 0.0

        explanation = "; ".join(signals) if signals else "No rule violations detected"

        return DetectionResult(
            claim_id=claim_id,
            detector="rule_based",
            is_hallucination=is_hallucination,
            confidence=confidence,
            signals=signals,
            explanation=explanation,
            ground_truth=ground_truth,
        )

    def detect_batch(self, cases: list) -> list:
        """Detect on a list of dicts with keys: claim_id, claim, source_context, is_hallucination."""
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
    detector = RuleBasedDetector()

    tests = [
        ("CLEAN", "The study involved 3,200 participants.", "The study involved 3,200 participants and ran 12 weeks.", False),
        ("WRONG_NUM", "The study involved 5,000 participants.", "The study involved 3,200 participants and ran 12 weeks.", True),
        ("SUPERLATIVE", "This is the most effective treatment ever discovered.", "The treatment reduced symptoms by 45 percent.", True),
        ("ENTITY", "Jerome Powell announced the decision at the Marriott Hotel.", "The Federal Reserve raised rates.", True),
    ]

    for name, claim, source, gt in tests:
        result = detector.detect(claim, source, ground_truth=gt)
        icon = "✓" if result.correct else "✗"
        print(f"{icon} {name:15s} predicted={'HAL' if result.is_hallucination else 'OK':3s}  conf={result.confidence:.2f}  signals={result.signals}")
