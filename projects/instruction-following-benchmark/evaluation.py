"""
evaluation.py
=============
Instruction Following Benchmark — Evaluation Engine
Project: P3 · prompt-engineering-lab

Scores model outputs against machine-checkable constraint rubrics.
Each constraint type returns pass/fail + failure reason.

Constraint types:
  step_present          — keyword(s) must appear (regex OR logic)
  exact_phrase          — exact string must appear verbatim
  word_absent           — word(s) must NOT appear
  char_absent           — character must NOT appear
  tone_word_present     — at least one tone word must appear
  tone_word_absent      — none of the tone words may appear
  numbered_list         — output must contain N numbered items
  paragraph_count       — output must have N paragraphs
  word_count_min        — output must have >= N words
  word_count_max        — output must have <= N words
  step_count            — keyword must appear >= N times
  allocation_sum        — percentages in output must sum to target
  contains_pattern      — regex pattern must match
  starts_with_caps_headline — first line must be ALL CAPS
  sentence_not_starts_with  — no sentence may start with given word
  not_starts_with_question  — output must not open with a question

Failure taxonomy (why did it fail?):
  MISSED_STEP           — required step/content not present
  VIOLATED_NEGATION     — used a forbidden word/phrase
  WRONG_FORMAT          — wrong structure (count, paragraphs, etc.)
  WRONG_TONE            — tone/persona not maintained
  LENGTH_VIOLATION      — too long or too short
  PARTIAL_COMPLIANCE    — passed some constraints, failed others
  FULL_COMPLIANCE       — all constraints passed
"""

import re
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class ConstraintResult:
    constraint_type: str
    passed: bool
    detail: str = ""          # human-readable explanation
    failure_mode: str = ""    # taxonomy label


@dataclass
class EvalResult:
    task_id: str
    category: str
    difficulty: str
    model: str
    output: str
    latency_s: float
    constraint_results: list = field(default_factory=list)
    # Aggregate scores
    constraints_total: int = 0
    constraints_passed: int = 0
    pass_rate: float = 0.0
    failure_modes: list = field(default_factory=list)
    failure_taxonomy: str = ""   # primary failure label
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Flatten constraint_results to JSON string for CSV
        d["constraint_results"] = json.dumps(
            [asdict(c) for c in self.constraint_results]
        )
        d["failure_modes"] = json.dumps(self.failure_modes)
        return d


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _words(text: str) -> list:
    return re.findall(r"\b\w+\b", text.lower())

def _word_count(text: str) -> int:
    return len(_words(text))

def _sentences(text: str) -> list:
    return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

def _paragraphs(text: str) -> list:
    return [p.strip() for p in re.split(r'\n\s*\n', text.strip()) if p.strip()]

def _extract_percentages(text: str) -> list:
    """Find all percentage numbers in text."""
    return [int(m) for m in re.findall(r'(\d+)\s*%', text)]


# ──────────────────────────────────────────────
# Constraint checkers
# ──────────────────────────────────────────────

def check_step_present(output: str, constraint: dict) -> ConstraintResult:
    keywords = constraint.get("keyword", "")
    pattern = re.compile(keywords, re.IGNORECASE)
    passed = bool(pattern.search(output))
    step_id = constraint.get("id", "?")
    return ConstraintResult(
        constraint_type="step_present",
        passed=passed,
        detail=f"Step {step_id}: keyword pattern '{keywords[:40]}' {'found' if passed else 'NOT FOUND'}",
        failure_mode="" if passed else "MISSED_STEP",
    )


def check_exact_phrase(output: str, constraint: dict) -> ConstraintResult:
    phrase = constraint.get("phrase", "")
    passed = phrase in output
    return ConstraintResult(
        constraint_type="exact_phrase",
        passed=passed,
        detail=f"Exact phrase '{phrase}' {'found' if passed else 'NOT FOUND'}",
        failure_mode="" if passed else "MISSED_STEP",
    )


def check_word_absent(output: str, constraint: dict) -> ConstraintResult:
    forbidden = constraint.get("words", [])
    violations = []
    for word in forbidden:
        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
        if pattern.search(output):
            violations.append(word)
    passed = len(violations) == 0
    return ConstraintResult(
        constraint_type="word_absent",
        passed=passed,
        detail=f"Forbidden words {'none used' if passed else 'FOUND: ' + ', '.join(violations)}",
        failure_mode="" if passed else "VIOLATED_NEGATION",
    )


def check_char_absent(output: str, constraint: dict) -> ConstraintResult:
    char = constraint.get("char", "")
    passed = char not in output
    return ConstraintResult(
        constraint_type="char_absent",
        passed=passed,
        detail=f"Forbidden character '{char}' {'absent' if passed else 'FOUND'}",
        failure_mode="" if passed else "VIOLATED_NEGATION",
    )


def check_tone_word_present(output: str, constraint: dict) -> ConstraintResult:
    words = constraint.get("words", [])
    # words can be a single pipe-delimited string or a list
    if isinstance(words, str):
        words = [words]
    found = []
    for word_or_pattern in words:
        if re.search(word_or_pattern, output, re.IGNORECASE):
            found.append(word_or_pattern)
    passed = len(found) > 0
    return ConstraintResult(
        constraint_type="tone_word_present",
        passed=passed,
        detail=f"Tone words {'found: ' + ', '.join(found[:3]) if passed else 'NONE FOUND from: ' + str(words[:3])}",
        failure_mode="" if passed else "WRONG_TONE",
    )


def check_tone_word_absent(output: str, constraint: dict) -> ConstraintResult:
    words = constraint.get("words", [])
    if isinstance(words, str):
        words = [words]
    violations = [w for w in words if re.search(w, output, re.IGNORECASE)]
    passed = len(violations) == 0
    return ConstraintResult(
        constraint_type="tone_word_absent",
        passed=passed,
        detail=f"Prohibited tone {'none found' if passed else 'FOUND: ' + ', '.join(violations[:3])}",
        failure_mode="" if passed else "WRONG_TONE",
    )


def check_numbered_list(output: str, constraint: dict) -> ConstraintResult:
    target = constraint.get("count", 1)
    # Match lines starting with number + period/paren
    matches = re.findall(r'^\s*\d+[\.\)]\s+\S', output, re.MULTILINE)
    actual = len(matches)
    passed = actual >= target
    return ConstraintResult(
        constraint_type="numbered_list",
        passed=passed,
        detail=f"Numbered items: found {actual}, required {target}",
        failure_mode="" if passed else "WRONG_FORMAT",
    )


def check_paragraph_count(output: str, constraint: dict) -> ConstraintResult:
    target = constraint.get("count", 1)
    actual = len(_paragraphs(output))
    passed = actual == target
    return ConstraintResult(
        constraint_type="paragraph_count",
        passed=passed,
        detail=f"Paragraphs: found {actual}, required {target}",
        failure_mode="" if passed else "WRONG_FORMAT",
    )


def check_word_count_min(output: str, constraint: dict) -> ConstraintResult:
    minimum = constraint.get("min", 0)
    actual = _word_count(output)
    passed = actual >= minimum
    return ConstraintResult(
        constraint_type="word_count_min",
        passed=passed,
        detail=f"Word count {actual} (min {minimum})",
        failure_mode="" if passed else "LENGTH_VIOLATION",
    )


def check_word_count_max(output: str, constraint: dict) -> ConstraintResult:
    maximum = constraint.get("max", 9999)
    actual = _word_count(output)
    passed = actual <= maximum
    return ConstraintResult(
        constraint_type="word_count_max",
        passed=passed,
        detail=f"Word count {actual} (max {maximum})",
        failure_mode="" if passed else "LENGTH_VIOLATION",
    )


def check_step_count(output: str, constraint: dict) -> ConstraintResult:
    keyword = constraint.get("keyword", "")
    min_count = constraint.get("min_count", 1)
    matches = re.findall(keyword, output, re.IGNORECASE)
    actual = len(matches)
    passed = actual >= min_count
    return ConstraintResult(
        constraint_type="step_count",
        passed=passed,
        detail=f"Keyword '{keyword[:30]}' appeared {actual} times (min {min_count})",
        failure_mode="" if passed else "MISSED_STEP",
    )


def check_allocation_sum(output: str, constraint: dict) -> ConstraintResult:
    target = constraint.get("target", 100)
    percentages = _extract_percentages(output)
    total = sum(percentages)
    passed = abs(total - target) <= 2  # 2% tolerance
    return ConstraintResult(
        constraint_type="allocation_sum",
        passed=passed,
        detail=f"Percentages found: {percentages} = {total}% (target {target}%)",
        failure_mode="" if passed else "WRONG_FORMAT",
    )


def check_contains_pattern(output: str, constraint: dict) -> ConstraintResult:
    pattern = constraint.get("pattern", "")
    passed = bool(re.search(pattern, output))
    return ConstraintResult(
        constraint_type="contains_pattern",
        passed=passed,
        detail=f"Pattern '{pattern}' {'found' if passed else 'NOT FOUND'}",
        failure_mode="" if passed else "MISSED_STEP",
    )


def check_starts_with_caps_headline(output: str, constraint: dict) -> ConstraintResult:
    first_line = output.strip().split("\n")[0].strip()
    # At least 3 words, mostly uppercase letters
    words_in_line = re.findall(r'[A-Za-z]+', first_line)
    caps_words = [w for w in words_in_line if w.isupper() and len(w) > 1]
    passed = len(words_in_line) >= 2 and len(caps_words) >= len(words_in_line) * 0.6
    return ConstraintResult(
        constraint_type="starts_with_caps_headline",
        passed=passed,
        detail=f"First line: '{first_line[:60]}' — {'ALL CAPS OK' if passed else 'NOT ALL CAPS'}",
        failure_mode="" if passed else "WRONG_FORMAT",
    )


def check_sentence_not_starts_with(output: str, constraint: dict) -> ConstraintResult:
    word = constraint.get("word", "")
    pattern = re.compile(r'(?:^|(?<=[.!?])\s+)' + re.escape(word) + r'\b', re.IGNORECASE | re.MULTILINE)
    violations = pattern.findall(output)
    passed = len(violations) == 0
    return ConstraintResult(
        constraint_type="sentence_not_starts_with",
        passed=passed,
        detail=f"Sentences starting with '{word}': {'none' if passed else str(len(violations)) + ' found'}",
        failure_mode="" if passed else "VIOLATED_NEGATION",
    )


def check_not_starts_with_question(output: str, constraint: dict) -> ConstraintResult:
    first_sentence = output.strip().split(".")[0].strip()
    passed = not first_sentence.endswith("?") and not output.strip().startswith("Are ") \
             and not output.strip().startswith("Is ") and not output.strip().startswith("Do ") \
             and not output.strip().startswith("Can ") and not output.strip().startswith("Have ") \
             and not output.strip().startswith("What ") and not output.strip().startswith("How ") \
             and not output.strip().startswith("Why ") and not output.strip().startswith("When ")
    return ConstraintResult(
        constraint_type="not_starts_with_question",
        passed=passed,
        detail=f"Opening is{'not ' if passed else ' '}a question",
        failure_mode="" if passed else "VIOLATED_NEGATION",
    )


# ──────────────────────────────────────────────
# Constraint dispatcher
# ──────────────────────────────────────────────

CHECKER_MAP = {
    "step_present":             check_step_present,
    "exact_phrase":             check_exact_phrase,
    "word_absent":              check_word_absent,
    "char_absent":              check_char_absent,
    "tone_word_present":        check_tone_word_present,
    "tone_word_absent":         check_tone_word_absent,
    "numbered_list":            check_numbered_list,
    "paragraph_count":          check_paragraph_count,
    "word_count_min":           check_word_count_min,
    "word_count_max":           check_word_count_max,
    "step_count":               check_step_count,
    "allocation_sum":           check_allocation_sum,
    "contains_pattern":         check_contains_pattern,
    "starts_with_caps_headline":check_starts_with_caps_headline,
    "sentence_not_starts_with": check_sentence_not_starts_with,
    "not_starts_with_question": check_not_starts_with_question,
}


# ──────────────────────────────────────────────
# Failure taxonomy
# ──────────────────────────────────────────────

def classify_failure(constraint_results: list, pass_rate: float) -> str:
    """
    Determine the primary failure mode from a set of constraint results.
    Returns a single taxonomy label.
    """
    if pass_rate == 1.0:
        return "FULL_COMPLIANCE"

    failed = [c for c in constraint_results if not c.passed]
    modes = [c.failure_mode for c in failed if c.failure_mode]

    if not modes:
        return "FULL_COMPLIANCE"

    # Primary failure = most common mode among failures
    from collections import Counter
    mode_counts = Counter(modes)
    primary = mode_counts.most_common(1)[0][0]

    # If multiple modes present, escalate to PARTIAL_COMPLIANCE
    if len(set(modes)) > 1:
        return f"PARTIAL_COMPLIANCE ({primary}+)"

    return primary


# ──────────────────────────────────────────────
# Master evaluation function
# ──────────────────────────────────────────────

def evaluate_output(
    task_id: str,
    category: str,
    difficulty: str,
    model: str,
    output: str,
    latency_s: float,
    constraints_json: str,
) -> EvalResult:
    """
    Evaluate a model's output against the task's constraint rubric.
    Returns a fully populated EvalResult.
    """
    result = EvalResult(
        task_id=task_id,
        category=category,
        difficulty=difficulty,
        model=model,
        output=output,
        latency_s=latency_s,
    )

    try:
        constraints = json.loads(constraints_json)
    except json.JSONDecodeError as e:
        result.error = f"Failed to parse constraints JSON: {e}"
        return result

    constraint_results = []
    for c in constraints:
        ctype = c.get("type", "")
        checker = CHECKER_MAP.get(ctype)
        if checker:
            cr = checker(output, c)
            constraint_results.append(cr)
        else:
            logger.warning(f"Unknown constraint type: {ctype}")

    result.constraint_results = constraint_results
    result.constraints_total  = len(constraint_results)
    result.constraints_passed = sum(1 for c in constraint_results if c.passed)
    result.pass_rate = round(
        result.constraints_passed / result.constraints_total, 4
    ) if result.constraints_total > 0 else 0.0

    result.failure_modes  = list({c.failure_mode for c in constraint_results if not c.passed})
    result.failure_taxonomy = classify_failure(constraint_results, result.pass_rate)

    return result


# ──────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    test_output = """1. Paris is the capital of France.
2. Paris has a population of approximately 2.1 million people in the city proper.
3. The Eiffel Tower is one of the most famous landmarks in Paris.
That concludes the summary."""

    constraints = json.dumps([
        {"type": "step_present", "id": 1, "keyword": "paris"},
        {"type": "step_present", "id": 2, "keyword": "population|million|people|inhabitants"},
        {"type": "step_present", "id": 3, "keyword": "eiffel|louvre|notre|landmark|tower"},
        {"type": "exact_phrase", "phrase": "That concludes the summary."},
    ])

    result = evaluate_output(
        task_id="MS02", category="multi_step", difficulty="easy",
        model="test", output=test_output, latency_s=1.0,
        constraints_json=constraints,
    )

    print(f"\n Task: MS02")
    print(f" Pass rate: {result.pass_rate:.0%} ({result.constraints_passed}/{result.constraints_total})")
    print(f" Taxonomy:  {result.failure_taxonomy}")
    for cr in result.constraint_results:
        icon = "✓" if cr.passed else "✗"
        print(f"   {icon} [{cr.constraint_type}] {cr.detail}")
