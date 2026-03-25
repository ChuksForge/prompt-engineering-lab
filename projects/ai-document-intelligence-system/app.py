"""
app.py
======
Document Intelligence System — Gradio Demo
Project: P9 · prompt-engineering-lab

Upload a document (TXT, PDF, DOCX, CSV) or paste text.
Pipeline runs: classify → extract → index → QA

Usage:
    pip install gradio
    python app.py
    # Opens at http://127.0.0.1:7860
"""

import os
import tempfile
from pathlib import Path

SAMPLE_TEXTS = {
    "📄 Service Contract": open("data/documents/contract_service_agreement.txt").read() if Path("data/documents/contract_service_agreement.txt").exists() else "",
    "💰 Invoice":          open("data/documents/invoice_design_services.txt").read()    if Path("data/documents/invoice_design_services.txt").exists() else "",
    "📊 Research Report":  open("data/documents/research_report_ai_productivity.txt").read()[:2000] + "..." if Path("data/documents/research_report_ai_productivity.txt").exists() else "",
    "📅 Meeting Minutes":  open("data/documents/meeting_minutes_q1_roadmap.txt").read() if Path("data/documents/meeting_minutes_q1_roadmap.txt").exists() else "",
    "📈 Financial Statement": open("data/documents/financial_statement_novatech.txt").read() if Path("data/documents/financial_statement_novatech.txt").exists() else "",
}

MODEL_CONFIG = {
    "GPT-4o-mini (OpenAI)":     ("openai",     "gpt-4o-mini"),
    "GPT-4o (OpenAI)":          ("openai",     "gpt-4o"),
    "Claude Haiku (Anthropic)": ("anthropic",  "claude-haiku-4-5-20251001"),
    "Claude Sonnet (Anthropic)":("anthropic",  "claude-sonnet-4-6"),
    "Mistral 7B (OpenRouter)":  ("openrouter", "mistralai/mistral-7b-instruct"),
}


def get_client(model_name):
    if model_name not in MODEL_CONFIG:
        return None, None, None
    provider, model_id = MODEL_CONFIG[model_name]
    key_map = {
        "openai":     "OPENAI_API_KEY",
        "anthropic":  "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    key = os.environ.get(key_map[provider], "")
    if not key:
        return None, provider, f"❌ {key_map[provider]} not set."
    try:
        if provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=key), provider, model_id
        else:
            from openai import OpenAI
            kwargs = {"api_key": key}
            if provider == "openrouter":
                kwargs["base_url"] = "https://openrouter.ai/api/v1"
            return OpenAI(**kwargs), provider, model_id
    except Exception as e:
        return None, provider, f"❌ {e}"


def process_document(text, model_name, question):
    if not text or not text.strip():
        return "⚠️ Please enter text or upload a document.", "", "", ""

    client, provider, model_id = get_client(model_name)
    if isinstance(model_id, str) and model_id.startswith("❌"):
        return model_id, "", "", ""
    if not client:
        return "❌ Could not initialize API client.", "", "", ""

    from intelligence import DocumentClassifier, DocumentExtractor, DocumentQA
    from chunker import Chunker, Indexer

    classifier = DocumentClassifier(client, provider, model_id)
    extractor  = DocumentExtractor(client, provider, model_id)
    chunker    = Chunker(chunk_size=200, chunk_overlap=40)

    # Classify
    clf = classifier.classify(text)
    clf_out = (
        f"**Document Type:** {clf.document_type.replace('_',' ').title()}  "
        f"(confidence: {clf.confidence:.0%})\n\n"
        f"**Summary:** {clf.summary}\n\n"
        f"**Routing Tags:** {', '.join(clf.routing_tags) if clf.routing_tags else 'none'}"
    )

    # Extract
    ext = extractor.extract(text)
    ext_lines = []
    if ext.entities:
        for etype, elist in ext.entities.items():
            if elist:
                ext_lines.append(f"**{etype.title()}:** {', '.join(str(e) for e in elist)}")
    if ext.dates:
        ext_lines.append("\n**Dates:**")
        for k, v in ext.dates.items():
            ext_lines.append(f"  • {k}: {v}")
    if ext.monetary_values:
        ext_lines.append("\n**Monetary Values:**")
        for mv in ext.monetary_values:
            ext_lines.append(f"  • {mv.get('label','')}: {mv.get('amount','')} {mv.get('currency','')}")
    if ext.key_facts:
        ext_lines.append("\n**Key Facts:**")
        for f_item in ext.key_facts[:6]:
            ext_lines.append(f"  • {f_item}")
    if ext.action_items:
        ext_lines.append("\n**Action Items:**")
        for ai in ext.action_items[:4]:
            ext_lines.append(f"  • {ai.get('action','')} → {ai.get('owner','')} by {ai.get('due_date','')}")

    ext_out = "\n".join(ext_lines) if ext_lines else "_No structured data extracted._"

    # Index + QA
    chunks  = chunker.chunk("doc", text)
    indexer = Indexer(use_embeddings=False)
    indexer.add_chunks(chunks)
    qa_engine = DocumentQA(client, provider, model_id, indexer=indexer)

    qa_out = ""
    if question and question.strip():
        qr = qa_engine.answer(question)
        qa_out = f"**Q:** {question}\n\n**A:** {qr.answer}"
        if qr.citations:
            qa_out += f"\n\n**Source excerpt:** _{qr.citations[0][:200]}..._"
    else:
        qa_out = "_Enter a question above to query the document._"

    stats = (
        f"📄 {len(chunks)} chunks · "
        f"{len(text.split()):,} words · "
        f"{clf.document_type.replace('_',' ').title()}"
    )

    return clf_out, ext_out, qa_out, stats


def handle_upload(file):
    if file is None:
        return ""
    try:
        from ingestion import ingest
        doc = ingest(file.name)
        return doc.text if not doc.error else f"Error reading file: {doc.error}"
    except Exception as e:
        return f"Error: {e}"


def build_app():
    import gradio as gr

    def load_sample(name):
        return SAMPLE_TEXTS.get(name, "")

    with gr.Blocks(title="AI Document Intelligence") as demo:
        gr.Markdown(
            "# 📑 AI Document Intelligence System\n"
            "**P9 · prompt-engineering-lab** — Classify, extract, and query any document"
        )

        with gr.Row():
            with gr.Column(scale=1):
                model_dd = gr.Dropdown(
                    choices=list(MODEL_CONFIG.keys()),
                    value="GPT-4o-mini (OpenAI)",
                    label="Model",
                )
                sample_dd = gr.Dropdown(
                    choices=list(SAMPLE_TEXTS.keys()),
                    label="Load a sample document",
                    value=None,
                )
                load_btn  = gr.Button("Load Sample →", size="sm")
                file_upload = gr.File(
                    label="Or upload a file (.txt, .pdf, .docx, .csv)",
                    file_types=[".txt",".pdf",".docx",".csv",".md"],
                )
                question_box = gr.Textbox(
                    label="Question to ask the document (optional)",
                    placeholder="e.g. What is the total amount due?",
                    lines=2,
                )
                process_btn = gr.Button("⚡ Process Document", variant="primary")

            with gr.Column(scale=2):
                doc_input = gr.Textbox(
                    label="Document Text",
                    placeholder="Paste document text here, or load a sample / upload a file...",
                    lines=16,
                )

        stats_out = gr.Markdown(value="")

        with gr.Tabs():
            with gr.Tab("🏷️ Classification"):
                clf_out = gr.Markdown()
            with gr.Tab("🔍 Extraction"):
                ext_out = gr.Markdown()
            with gr.Tab("💬 Document QA"):
                qa_out = gr.Markdown()

        gr.Markdown(
            "**Pipeline:** Classify → Extract entities/dates/values/facts → "
            "Chunk & index → Answer questions with citations  \n"
            "**Supports:** .txt · .pdf (requires pypdf) · .docx (requires python-docx) · .csv"
        )

        load_btn.click(fn=load_sample, inputs=sample_dd, outputs=doc_input)
        sample_dd.change(fn=load_sample, inputs=sample_dd, outputs=doc_input)
        file_upload.change(fn=handle_upload, inputs=file_upload, outputs=doc_input)
        process_btn.click(
            fn=process_document,
            inputs=[doc_input, model_dd, question_box],
            outputs=[clf_out, ext_out, qa_out, stats_out],
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
