"""
pipeline.py
===========
Document Intelligence System — Pipeline Orchestrator
Project: P9 · prompt-engineering-lab

Full pipeline: ingest → classify → chunk → index → extract → QA → benchmark

Usage:
    python pipeline.py                         # process all documents
    python pipeline.py --doc contract_service_agreement.txt
    python pipeline.py --quick                 # 2 docs, skip benchmark
    python pipeline.py --models openai
    python pipeline.py --no-benchmark          # skip accuracy scoring
    python pipeline.py --use-embeddings        # use sentence-transformers
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path

import pandas as pd

from ingestion  import ingest_directory, DocumentRaw
from chunker    import Chunker, Indexer
from intelligence import DocumentClassifier, DocumentExtractor, DocumentQA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
DATA_DIR    = Path("data")
GT_PATH     = DATA_DIR / "ground_truth" / "ground_truth.json"

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
        "google/gemini-2.0-flash-001":  {"provider": "openrouter", "label": "Gemini 2.0 Flash"},
    },
}


# ── Clients ──────────────────────────────────────────────────

def init_client(model_filter):
    for grp, grp_models in MODELS.items():
        if model_filter and grp not in model_filter:
            continue
        for model_id, mmeta in grp_models.items():
            provider = mmeta["provider"]
            env_key  = {
                "openai":     "OPENAI_API_KEY",
                "anthropic":  "ANTHROPIC_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
            }[provider]
            if not os.environ.get(env_key):
                continue
            try:
                if provider == "anthropic":
                    import anthropic
                    client = anthropic.Anthropic(api_key=os.environ[env_key])
                else:
                    from openai import OpenAI
                    kwargs = {"api_key": os.environ[env_key]}
                    if provider == "openrouter":
                        kwargs["base_url"] = "https://openrouter.ai/api/v1"
                    client = OpenAI(**kwargs)
                logger.info(f"  Using: {mmeta['label']} ({provider})")
                return client, provider, model_id, mmeta["label"]
            except Exception as e:
                logger.warning(f"  {mmeta['label']}: {e}")
    raise RuntimeError("No API client available. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY.")


# ── Benchmark scoring ────────────────────────────────────────

def score_extraction(extracted, ground_truth: dict) -> dict:
    """Compare extracted fields against ground truth. Returns accuracy scores."""
    scores = {}

    # Document type classification
    gt_type = ground_truth.get("document_type", "")
    pred_type = extracted.get("classification", {}).get("document_type", "")
    scores["classification"] = 1.0 if pred_type == gt_type else 0.0

    # Entity recall
    gt_entities = []
    for v in ground_truth.get("entities", {}).values():
        if isinstance(v, list):
            gt_entities.extend([e.lower() for e in v])
        else:
            gt_entities.append(str(v).lower())

    pred_entities = []
    ext_ents = extracted.get("extraction", {}).get("entities", {})
    for v in ext_ents.values():
        if isinstance(v, list):
            pred_entities.extend([e.lower() for e in v])
        elif isinstance(v, str):
            pred_entities.append(v.lower())

    if gt_entities:
        hits = sum(1 for ge in gt_entities if any(ge in pe or pe in ge for pe in pred_entities))
        scores["entity_recall"] = round(hits / len(gt_entities), 3)
    else:
        scores["entity_recall"] = 1.0

    # Date extraction
    gt_dates = {k: v.lower() for k, v in ground_truth.get("dates", {}).items()}
    pred_dates_raw = extracted.get("extraction", {}).get("dates", {})
    pred_dates = {k: str(v).lower() for k, v in pred_dates_raw.items()}

    if gt_dates:
        date_hits = 0
        for gt_k, gt_v in gt_dates.items():
            for pk, pv in pred_dates.items():
                if any(part in pv for part in gt_v.split()) or gt_v in pv:
                    date_hits += 1
                    break
        scores["date_recall"] = round(date_hits / len(gt_dates), 3)
    else:
        scores["date_recall"] = 1.0

    # QA accuracy
    qa_results = extracted.get("qa", [])
    gt_qa = ground_truth.get("qa_pairs", [])
    if qa_results and gt_qa:
        qa_hits = 0
        for qa_r in qa_results:
            pred_answer = qa_r.get("answer", "").lower()
            # Find matching GT question
            for gt_q in gt_qa:
                if gt_q["question"].lower() == qa_r.get("question", "").lower():
                    gt_answer = gt_q["answer"].lower()
                    # Check if any key token from GT answer appears in predicted
                    gt_tokens = [t for t in gt_answer.split() if len(t) > 3]
                    if gt_tokens and any(t in pred_answer for t in gt_tokens):
                        qa_hits += 1
                    break
        scores["qa_accuracy"] = round(qa_hits / len(qa_results), 3) if qa_results else 0.0
    else:
        scores["qa_accuracy"] = 0.0

    # Composite
    scores["composite"] = round(
        0.2 * scores["classification"] +
        0.3 * scores["entity_recall"] +
        0.2 * scores["date_recall"] +
        0.3 * scores["qa_accuracy"], 3
    )
    return scores


# ── Main pipeline ────────────────────────────────────────────

def process_document(doc: DocumentRaw, classifier, extractor, chunker, use_embeddings, qa_questions=None):
    """Process one document through the full pipeline."""
    result = {
        "doc_id":    doc.filename.replace(".txt","").replace(".pdf","").replace(".docx",""),
        "filename":  doc.filename,
        "word_count": doc.word_count,
        "classification": {},
        "extraction": {},
        "qa": [],
        "chunks": 0,
        "latency_s": {},
    }

    if doc.error or not doc.text:
        result["error"] = doc.error or "Empty document"
        return result

    # Classification
    t0 = time.time()
    clf = classifier.classify(doc.text, result["doc_id"])
    result["classification"]   = clf.to_dict()
    result["latency_s"]["classify"] = round(time.time() - t0, 2)
    logger.info(f"    Type: {clf.document_type} ({clf.confidence:.0%}) — {clf.summary[:60]}")

    # Extraction
    t0 = time.time()
    ext = extractor.extract(doc.text, result["doc_id"])
    result["extraction"]       = ext.to_dict()
    result["latency_s"]["extract"] = round(time.time() - t0, 2)
    n_facts = len(ext.key_facts)
    n_money = len(ext.monetary_values)
    logger.info(f"    Extracted: {n_facts} key facts, {n_money} monetary values")

    # Chunking + indexing
    t0 = time.time()
    chunks  = chunker.chunk(result["doc_id"], doc.text)
    indexer = Indexer(use_embeddings=use_embeddings)
    indexer.add_chunks(chunks)
    result["chunks"]            = len(chunks)
    result["latency_s"]["index"] = round(time.time() - t0, 2)
    logger.info(f"    Chunks: {len(chunks)}")

    # QA
    if qa_questions:
        t0 = time.time()
        qa_engine = DocumentQA(
            client   = classifier.client,
            provider = classifier.provider,
            model    = classifier.model,
            indexer  = indexer,
        )
        # Temporarily attach client info to qa_engine
        qa_engine.client   = classifier.client
        qa_engine.provider = classifier.provider
        qa_engine.model    = classifier.model

        qa_results = []
        for q in qa_questions:
            qr = qa_engine.answer(q)
            qa_results.append(qr.to_dict())
        result["qa"]               = qa_results
        result["latency_s"]["qa"]  = round(time.time() - t0, 2)
        logger.info(f"    QA: {len(qa_results)} questions answered")

    return result


def run_pipeline(
    model_filter=None,
    doc_filter=None,
    quick=False,
    run_benchmark=True,
    use_embeddings=False,
):
    RESULTS_DIR.mkdir(exist_ok=True)

    client, provider, model_id, model_label = init_client(model_filter)

    classifier = DocumentClassifier(client, provider, model_id)
    extractor  = DocumentExtractor(client, provider, model_id)
    chunker    = Chunker(chunk_size=200, chunk_overlap=40)

    # Load documents
    docs = ingest_directory(str(DATA_DIR / "documents"))
    if doc_filter:
        docs = [d for d in docs if d.filename in doc_filter]
    if quick:
        docs = docs[:2]
        logger.info("Quick mode: 2 documents")

    # Load ground truth
    ground_truth = {}
    if GT_PATH.exists():
        with open(GT_PATH) as f:
            ground_truth = json.load(f)

    all_results = []
    benchmark_rows = []

    for doc in docs:
        doc_id = doc.filename.replace(".txt","").replace(".pdf","").replace(".docx","")
        logger.info(f"\n  Processing: {doc.filename} ({doc.word_count} words)")

        # Get QA questions from ground truth
        qa_questions = []
        if doc_id in ground_truth:
            qa_questions = [qp["question"] for qp in ground_truth[doc_id].get("qa_pairs", [])]

        result = process_document(doc, classifier, extractor, chunker, use_embeddings, qa_questions)
        result["model"] = model_label
        all_results.append(result)

        # Benchmark
        if run_benchmark and doc_id in ground_truth:
            scores = score_extraction(result, ground_truth[doc_id])
            benchmark_rows.append({
                "doc_id":    doc_id,
                "model":     model_label,
                **scores,
                "total_latency": sum(result["latency_s"].values()),
            })
            logger.info(
                f"    Accuracy: composite={scores['composite']:.3f}  "
                f"classification={scores['classification']:.0f}  "
                f"entity_recall={scores['entity_recall']:.3f}  "
                f"qa_accuracy={scores['qa_accuracy']:.3f}"
            )

        time.sleep(0.5)

    # Save results
    with open(RESULTS_DIR / "pipeline_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\n  pipeline_results.json saved ({len(all_results)} documents)")

    if benchmark_rows:
        bdf = pd.DataFrame(benchmark_rows)
        bdf.to_csv(RESULTS_DIR / "benchmark_results.csv", index=False)
        logger.info("  benchmark_results.csv saved")
        logger.info(f"\n  OVERALL: composite={bdf['composite'].mean():.3f}  qa={bdf['qa_accuracy'].mean():.3f}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",         type=str)
    parser.add_argument("--doc",            type=str)
    parser.add_argument("--quick",          action="store_true")
    parser.add_argument("--no-benchmark",   action="store_true")
    parser.add_argument("--use-embeddings", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        model_filter    = args.models.split(",") if args.models else None,
        doc_filter      = [args.doc]             if args.doc    else None,
        quick           = args.quick,
        run_benchmark   = not args.no_benchmark,
        use_embeddings  = args.use_embeddings,
    )
