"""detectors package — hallucination detection strategies"""
from .rule_based  import RuleBasedDetector
from .llm_judge   import LLMJudgeDetector
from .entailment  import EntailmentDetector

__all__ = ["RuleBasedDetector", "LLMJudgeDetector", "EntailmentDetector"]
