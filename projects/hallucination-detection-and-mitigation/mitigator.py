"""
mitigator.py
============
Hallucination Detection & Mitigation — Mitigation Pipeline
Project: P8 · prompt-engineering-lab

Three mitigation strategies:
  1. grounded_rewrite    — rewrite with explicit grounding instructions
  2. self_critique       — ask model to critique then correct its own output
  3. citation_enforced   — force model to cite every claim with a source quote

Each strategy takes (claim, source) → corrected_claim + re-scored detection result.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MitigationResult:
    claim_id: str
    strategy: str
    original_claim: str
    mitigated_claim: str
    original_score: float      # detector confidence pre-mitigation
    mitigated_score: float     # detector confidence post-mitigation
    improvement: float         # original - mitigated (positive = improved)
    still_hallucinating: bool
    latency_s: float
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "claim_id":          self.claim_id,
            "strategy":          self.strategy,
            "original_claim":    self.original_claim,
            "mitigated_claim":   self.mitigated_claim,
            "original_score":    self.original_score,
            "mitigated_score":   self.mitigated_score,
            "improvement":       round(self.improvement, 4),
            "still_hallucinating": self.still_hallucinating,
            "latency_s":         round(self.latency_s, 3),
            "error":             self.error,
        }


# ── Prompt templates ─────────────────────────────────────────

GROUNDED_REWRITE_PROMPT = """The following claim may contain inaccuracies. Rewrite it to be completely faithful to the source text provided. Only include information that is explicitly stated in the source.

SOURCE TEXT:
{source}

ORIGINAL CLAIM (may be inaccurate):
{claim}

REWRITTEN CLAIM (strictly grounded in source, fix any inaccuracies):"""


SELF_CRITIQUE_PROMPT = """You will review a claim for factual accuracy against a source, then provide a corrected version.

SOURCE TEXT:
{source}

CLAIM:
{claim}

Step 1 — Identify any facts in the claim that are NOT supported by or contradict the source:
[critique]

Step 2 — Write a corrected version that is fully faithful to the source (or keep the original if accurate):
CORRECTED CLAIM:"""


CITATION_ENFORCED_PROMPT = """Rewrite the following claim so that every factual statement is directly traceable to the source text. Format your response as:

CLAIM: [the rewritten claim, using only facts from the source]
SUPPORT: "[exact quote from source that supports each key fact]"

If the original claim contains facts not in the source, remove or correct them.

SOURCE TEXT:
{source}

ORIGINAL CLAIM:
{claim}"""


# ── Mitigator ────────────────────────────────────────────────

class Mitigator:
    """
    Applies mitigation strategies to hallucinated claims.

    Args:
        client:    API client (OpenAI-compatible or Anthropic)
        model:     Model to use for rewriting
        provider:  "openai" | "anthropic" | "openrouter"
        detector:  A detector instance to re-score after mitigation
    """

    def __init__(self, client, model: str, provider: str, detector=None):
        self.client   = client
        self.model    = model
        self.provider = provider
        self.detector = detector

    def _call(self, prompt: str) -> tuple:
        import time
        t0 = time.time()
        try:
            if self.provider == "anthropic":
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=400,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text.strip(), time.time() - t0, None
            else:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=400,
                )
                return resp.choices[0].message.content.strip(), time.time() - t0, None
        except Exception as e:
            return "", time.time() - t0, str(e)

    def _extract_corrected(self, raw: str, strategy: str) -> str:
        """Extract the corrected claim from structured strategy outputs."""
        if strategy == "self_critique":
            if "CORRECTED CLAIM:" in raw:
                return raw.split("CORRECTED CLAIM:")[-1].strip()
        elif strategy == "citation_enforced":
            if "CLAIM:" in raw:
                lines = raw.split("\n")
                for line in lines:
                    if line.startswith("CLAIM:"):
                        return line.replace("CLAIM:", "").strip()
        return raw.strip()

    def _rescore(self, claim: str, source: str, claim_id: str) -> float:
        """Re-score with detector after mitigation. Returns confidence (0=clean, 1=still bad)."""
        if self.detector is None:
            return 0.0
        result = self.detector.detect(claim=claim, source=source, claim_id=claim_id)
        return result.confidence if result.is_hallucination else 0.0

    def mitigate(
        self,
        claim: str,
        source: str,
        original_confidence: float,
        claim_id: str = "unknown",
        strategy: str = "grounded_rewrite",
    ) -> MitigationResult:
        """
        Apply one mitigation strategy.

        Args:
            claim:               The (possibly hallucinated) claim
            source:              The source context
            original_confidence: Pre-mitigation detector confidence
            claim_id:            Identifier for tracking
            strategy:            "grounded_rewrite" | "self_critique" | "citation_enforced"
        """
        templates = {
            "grounded_rewrite":   GROUNDED_REWRITE_PROMPT,
            "self_critique":      SELF_CRITIQUE_PROMPT,
            "citation_enforced":  CITATION_ENFORCED_PROMPT,
        }
        template = templates.get(strategy, GROUNDED_REWRITE_PROMPT)
        prompt   = template.format(source=source, claim=claim)

        raw_output, latency, error = self._call(prompt)

        if error or not raw_output:
            return MitigationResult(
                claim_id=claim_id, strategy=strategy,
                original_claim=claim, mitigated_claim=claim,
                original_score=original_confidence, mitigated_score=original_confidence,
                improvement=0.0, still_hallucinating=True,
                latency_s=latency, error=error,
            )

        corrected        = self._extract_corrected(raw_output, strategy)
        mitigated_score  = self._rescore(corrected, source, claim_id)
        still_hal        = mitigated_score > 0.3
        improvement      = original_confidence - mitigated_score

        return MitigationResult(
            claim_id=claim_id,
            strategy=strategy,
            original_claim=claim,
            mitigated_claim=corrected,
            original_score=original_confidence,
            mitigated_score=mitigated_score,
            improvement=improvement,
            still_hallucinating=still_hal,
            latency_s=latency,
        )

    def mitigate_all_strategies(
        self,
        claim: str,
        source: str,
        original_confidence: float,
        claim_id: str = "unknown",
    ) -> list:
        """Run all three mitigation strategies and return comparison."""
        results = []
        for strategy in ["grounded_rewrite", "self_critique", "citation_enforced"]:
            result = self.mitigate(claim, source, original_confidence, claim_id, strategy)
            results.append(result)
        return results
