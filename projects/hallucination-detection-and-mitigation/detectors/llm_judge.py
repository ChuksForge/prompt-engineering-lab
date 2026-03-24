"""
detectors/llm_judge.py
=======================
Hallucination Detection — LLM-as-Judge Detector
Project: P8 · prompt-engineering-lab

Uses an LLM to assess whether a claim is faithful to its source context.
Higher recall than rule-based — catches subtle semantic hallucinations.
Costs API tokens. Configurable judge model.

Returns:
  - faithfulness score (0-5)
  - is_hallucination (True if score < threshold)
  - explanation
  - hallucination_type classification
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from .rule_based import DetectionResult

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are an expert fact-checker evaluating whether a claim is faithful to a source text.

SOURCE TEXT:
{source}

CLAIM TO EVALUATE:
{claim}

Evaluate the claim on these dimensions:

1. FAITHFULNESS (1-5): Is every factual statement in the claim supported by the source?
   1 = Contains major fabrications or contradictions
   2 = Contains significant unsupported claims
   3 = Mostly faithful but has minor unsupported details
   4 = Faithful with trivial or paraphrasing differences
   5 = Perfectly faithful, all facts traceable to source

2. HALLUCINATION_TYPE: If unfaithful, classify as one of:
   - fabricated_fact (specific wrong numbers, dates, names)
   - unsupported_claim (claim goes beyond what source states)
   - entity_invention (introduces people/places not in source)
   - contradiction (directly contradicts source)
   - none (no hallucination)

Respond ONLY with this JSON, no other text:
{{
  "faithfulness": <1-5>,
  "hallucination_type": "<type>",
  "is_hallucination": <true|false>,
  "confidence": <0.0-1.0>,
  "explanation": "<one sentence>"
}}"""


class LLMJudgeDetector:
    """
    LLM-based hallucination detector.

    Args:
        client:         An initialized PromptLabClient or direct API client.
        judge_model:    Model label to use for judging (default: gpt-4o-mini).
        threshold:      Faithfulness score below which claim is flagged (default: 3).
        provider:       "openai" | "anthropic" | "openrouter"
    """

    def __init__(
        self,
        client=None,
        judge_model: str = "gpt-4o-mini",
        provider: str = "openai",
        threshold: float = 3.0,
    ):
        self.client      = client
        self.judge_model = judge_model
        self.provider    = provider
        self.threshold   = threshold

    def _call_judge(self, prompt: str) -> dict:
        """Call the judge model and parse JSON response."""
        try:
            if self.provider == "anthropic":
                resp = self.client.messages.create(
                    model=self.judge_model,
                    max_tokens=300,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text.strip()
            else:  # openai or openrouter
                resp = self.client.chat.completions.create(
                    model=self.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=300,
                )
                raw = resp.choices[0].message.content.strip()

            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            return json.loads(raw)

        except json.JSONDecodeError as e:
            logger.warning(f"LLM judge JSON parse failed: {e}")
            return {}
        except Exception as e:
            logger.warning(f"LLM judge call failed: {e}")
            return {}

    def detect(
        self,
        claim: str,
        source: str,
        claim_id: str = "unknown",
        ground_truth: Optional[bool] = None,
    ) -> DetectionResult:

        prompt = JUDGE_PROMPT.format(source=source[:3000], claim=claim)
        data   = self._call_judge(prompt)

        if not data:
            return DetectionResult(
                claim_id=claim_id,
                detector="llm_judge",
                is_hallucination=False,
                confidence=0.0,
                explanation="Judge call failed",
                ground_truth=ground_truth,
            )

        faithfulness     = float(data.get("faithfulness", 3))
        is_hallucination = bool(data.get("is_hallucination", faithfulness < self.threshold))
        confidence       = float(data.get("confidence", abs(faithfulness - self.threshold) / 5))
        hall_type        = data.get("hallucination_type", "none")
        explanation      = data.get("explanation", "")

        return DetectionResult(
            claim_id=claim_id,
            detector="llm_judge",
            is_hallucination=is_hallucination,
            confidence=round(confidence, 3),
            signals=[f"faithfulness_score:{faithfulness}", f"type:{hall_type}"],
            explanation=f"[faithfulness={faithfulness}/5] {explanation}",
            ground_truth=ground_truth,
        )

    def detect_batch(self, cases: list) -> list:
        results = []
        for c in cases:
            result = self.detect(
                claim=c["claim"],
                source=c["source_context"],
                claim_id=c["claim_id"],
                ground_truth=bool(c.get("is_hallucination")),
            )
            results.append(result)
        return results
