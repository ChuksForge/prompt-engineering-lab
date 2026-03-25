"""
intelligence.py
===============
Document Intelligence System — Classifier, Extractor, QA
Project: P9 · prompt-engineering-lab

Three LLM-powered capabilities:
  DocumentClassifier — classifies document type and extracts routing metadata
  DocumentExtractor  — structured extraction: entities, dates, money, key facts
  DocumentQA         — retrieval-augmented QA with source citations
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
class ClassificationResult:
    document_type: str          # contract, invoice, report, minutes, financial, technical, other
    confidence: float           # 0-1
    secondary_type: str = ""    # if ambiguous
    routing_tags: list = field(default_factory=list)  # e.g. ["legal", "finance", "urgent"]
    summary: str = ""           # 1-sentence document summary
    language: str = "English"

    def to_dict(self): return asdict(self)


@dataclass
class ExtractionResult:
    document_id: str
    entities: dict = field(default_factory=dict)   # people, orgs, locations
    dates: dict = field(default_factory=dict)
    monetary_values: list = field(default_factory=list)
    key_facts: list = field(default_factory=list)
    action_items: list = field(default_factory=list)
    tables: list = field(default_factory=list)
    raw_json: dict = field(default_factory=dict)

    def to_dict(self): return asdict(self)


@dataclass
class QAResult:
    question: str
    answer: str
    citations: list = field(default_factory=list)  # list of source chunk texts
    confidence: float = 0.0
    answerable: bool = True

    def to_dict(self): return asdict(self)


# ──────────────────────────────────────────────
# LLM caller
# ──────────────────────────────────────────────

def _call(client, provider: str, model: str, prompt: str,
          max_tokens: int = 800, temperature: float = 0.1) -> str:
    try:
        if provider == "anthropic":
            resp = client.messages.create(
                model=model, max_tokens=max_tokens, temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return ""

def _parse_json(text: str) -> dict:
    try:
        clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        return json.loads(clean)
    except Exception:
        return {}


# ──────────────────────────────────────────────
# Document Classifier
# ──────────────────────────────────────────────

CLASSIFY_PROMPT = """Analyze the following document and classify it.

DOCUMENT (first 2000 characters):
{text}

Respond ONLY with this JSON:
{{
  "document_type": "<contract|invoice|research_report|meeting_minutes|financial_statement|technical_spec|email|other>",
  "confidence": <0.0-1.0>,
  "secondary_type": "<secondary type if ambiguous, else empty string>",
  "routing_tags": ["<tag1>", "<tag2>"],
  "summary": "<one sentence describing what this document is>",
  "language": "<language name>"
}}

Document types:
- contract: legal agreements, service agreements, NDAs, leases
- invoice: bills, purchase orders, receipts
- research_report: academic or business research, analysis reports
- meeting_minutes: notes from meetings, decisions, action items
- financial_statement: balance sheets, income statements, financial reports
- technical_spec: API docs, technical requirements, architecture docs
- email: email correspondence
- other: anything else"""


class DocumentClassifier:
    def __init__(self, client, provider: str, model: str):
        self.client   = client
        self.provider = provider
        self.model    = model

    def classify(self, text: str, doc_id: str = "doc") -> ClassificationResult:
        prompt = CLASSIFY_PROMPT.format(text=text[:2000])
        raw    = _call(self.client, self.provider, self.model, prompt, max_tokens=300)
        data   = _parse_json(raw)

        if not data:
            # Rule-based fallback
            return self._rule_based_classify(text)

        return ClassificationResult(
            document_type  = data.get("document_type", "other"),
            confidence     = float(data.get("confidence", 0.5)),
            secondary_type = data.get("secondary_type", ""),
            routing_tags   = data.get("routing_tags", []),
            summary        = data.get("summary", ""),
            language       = data.get("language", "English"),
        )

    def _rule_based_classify(self, text: str) -> ClassificationResult:
        """Zero-cost classification fallback using keyword signals."""
        text_lower = text.lower()
        signals = {
            "contract":            sum(1 for w in ["agreement","party","clause","shall","governing law","termination"] if w in text_lower),
            "invoice":             sum(1 for w in ["invoice","total due","payment terms","subtotal","line item","net 30"] if w in text_lower),
            "research_report":     sum(1 for w in ["methodology","findings","participants","study","conclusion","abstract"] if w in text_lower),
            "meeting_minutes":     sum(1 for w in ["attendees","action items","agenda","minutes","meeting","decision"] if w in text_lower),
            "financial_statement": sum(1 for w in ["revenue","assets","liabilities","equity","balance sheet","income"] if w in text_lower),
        }
        best = max(signals, key=signals.get)
        confidence = min(0.85, signals[best] * 0.15)
        return ClassificationResult(
            document_type=best if signals[best] > 0 else "other",
            confidence=confidence,
            routing_tags=[],
            summary="[Rule-based classification]",
        )


# ──────────────────────────────────────────────
# Document Extractor
# ──────────────────────────────────────────────

EXTRACT_PROMPT = """Extract structured information from the following document.

DOCUMENT:
{text}

Return ONLY this JSON (use empty lists/dicts if not applicable):
{{
  "entities": {{
    "people": ["<name>", ...],
    "organizations": ["<org>", ...],
    "locations": ["<location>", ...]
  }},
  "dates": {{
    "<date_label>": "<date_value>",
    ...
  }},
  "monetary_values": [
    {{"label": "<what it is>", "amount": "<value>", "currency": "USD"}},
    ...
  ],
  "key_facts": [
    "<important fact 1>",
    "<important fact 2>",
    ...
  ],
  "action_items": [
    {{"action": "<what to do>", "owner": "<who>", "due_date": "<when>"}},
    ...
  ]
}}"""


class DocumentExtractor:
    def __init__(self, client, provider: str, model: str):
        self.client   = client
        self.provider = provider
        self.model    = model

    def extract(self, text: str, doc_id: str = "doc") -> ExtractionResult:
        # For long documents, extract in sections
        if len(text) > 6000:
            text_for_extraction = text[:3000] + "\n...[middle omitted]...\n" + text[-1500:]
        else:
            text_for_extraction = text

        prompt = EXTRACT_PROMPT.format(text=text_for_extraction)
        raw    = _call(self.client, self.provider, self.model, prompt, max_tokens=1000)
        data   = _parse_json(raw)

        if not data:
            return ExtractionResult(document_id=doc_id)

        return ExtractionResult(
            document_id    = doc_id,
            entities       = data.get("entities", {}),
            dates          = data.get("dates", {}),
            monetary_values = data.get("monetary_values", []),
            key_facts      = data.get("key_facts", []),
            action_items   = data.get("action_items", []),
            raw_json       = data,
        )


# ──────────────────────────────────────────────
# Document QA
# ──────────────────────────────────────────────

QA_PROMPT = """Answer the question using ONLY the provided context from the document.
If the answer is not in the context, say: "Not found in document."
Always cite which source section your answer comes from.

CONTEXT:
{context}

QUESTION: {question}

Answer (with source reference):"""


class DocumentQA:
    def __init__(self, client, provider: str, model: str, indexer=None):
        self.client   = client
        self.provider = provider
        self.model    = model
        self.indexer  = indexer

    def answer(self, question: str, top_k: int = 4) -> QAResult:
        if not self.indexer or self.indexer.chunk_count == 0:
            return QAResult(
                question=question,
                answer="No document indexed. Call pipeline.process() first.",
                answerable=False,
            )

        context = self.indexer.retrieve_as_context(question, top_k=top_k)
        prompt  = QA_PROMPT.format(context=context, question=question)
        answer  = _call(self.client, self.provider, self.model, prompt, max_tokens=400)

        # Extract citations
        chunks   = self.indexer.retrieve(question, top_k=top_k)
        citations = [rc.chunk.text[:200] for rc in chunks[:2]]

        answerable = "not found in document" not in answer.lower()

        return QAResult(
            question  = question,
            answer    = answer,
            citations = citations,
            confidence= 0.9 if answerable else 0.1,
            answerable= answerable,
        )

    def answer_batch(self, questions: list) -> list:
        return [self.answer(q) for q in questions]
