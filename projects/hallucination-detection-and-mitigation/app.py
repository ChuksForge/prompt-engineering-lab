"""
app.py
======
Hallucination Detection & Mitigation — Gradio Demo
Project: P8 · prompt-engineering-lab

Usage:
    pip install gradio
    python app.py
    # Opens at http://127.0.0.1:7860
"""

import os
import re
import time

SAMPLE_PAIRS = {
    "✅ Accurate claim": (
        "The James Webb Space Telescope launched on December 25, 2021. It operates at the L2 Lagrange point, 1.5 million kilometers from Earth.",
        "The James Webb Space Telescope launched on December 25, 2021 and operates 1.5 million kilometers from Earth at the L2 point.",
    ),
    "❌ Wrong number": (
        "A clinical trial involving 3,200 participants found that a 12-week exercise program reduced depression symptoms by 45 percent.",
        "The 8-week exercise program with 5,000 participants reduced depression symptoms by 45 percent.",
    ),
    "❌ Unsupported superlative": (
        "The Federal Reserve raised rates 11 times between 2022 and 2023, bringing rates to 5.25-5.5 percent.",
        "The Fed's historic rate-hiking campaign was the most aggressive monetary tightening in modern history.",
    ),
    "❌ Context leak (invents info)": (
        "Sodium-ion batteries developed at MIT show 40 percent lower manufacturing cost and 35 percent faster charging.",
        "The MIT team has already secured partnerships with Tesla and CATL for commercial production.",
    ),
    "❌ Contradicts source": (
        "Sodium-ion batteries show 285 Wh/kg energy density — approximately 20 percent lower than current lithium-ion batteries.",
        "The MIT sodium-ion batteries have higher energy density than lithium-ion batteries.",
    ),
}


def _extract_numbers(text):
    return set(float(m) for m in re.findall(r'\b\d+(?:\.\d+)?\b', text))

def _extract_entities(text):
    return set(re.findall(r'\b[A-Z][a-z]{2,}\b', text))

SUPERLATIVES = re.compile(
    r'\b(most|best|worst|largest|smallest|highest|lowest|first|only|never|always|unprecedented|proven)\b',
    re.IGNORECASE
)

def rule_scan(source, claim):
    signals = []
    claim_nums  = _extract_numbers(claim)
    source_nums = _extract_numbers(source)

    for num in claim_nums:
        if num < 1:
            continue
        matched = any(abs(num - sn) / max(abs(sn), 1) <= 0.05 for sn in source_nums)
        if not matched and source_nums:
            signals.append(f"🔢 Number **{num}** not found in source")

    claim_entities  = _extract_entities(claim)
    source_entities = _extract_entities(source)
    exclude = {"The","This","That","These","Those","It","He","She","They","We","A","An"}
    invented = claim_entities - source_entities - exclude
    if len(invented) >= 2:
        signals.append(f"👤 Entities not in source: {', '.join(list(invented)[:4])}")

    if SUPERLATIVES.search(claim) and not SUPERLATIVES.search(source):
        match = SUPERLATIVES.search(claim)
        signals.append(f"⚠️ Unsupported superlative: **{match.group()}**")

    return signals


def llm_scan(source, claim, model_name, provider, model_id, client):
    prompt = f"""Evaluate whether this claim is faithful to the source text.

SOURCE: {source}

CLAIM: {claim}

Rate faithfulness 1-5 (5=perfectly faithful) and explain any issues.
Format: SCORE: X/5 | VERDICT: [faithful/hallucination] | REASON: [brief explanation]"""

    try:
        t0 = time.time()
        if provider == "anthropic":
            resp = client.messages.create(
                model=model_id, max_tokens=200, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            output = resp.content[0].text.strip()
        else:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=200,
            )
            output = resp.choices[0].message.content.strip()
        latency = time.time() - t0
        return output, latency
    except Exception as e:
        return f"API error: {e}", 0.0


MODEL_CONFIG = {
    "GPT-4o-mini (OpenAI)":     ("openai",     "gpt-4o-mini"),
    "GPT-4o (OpenAI)":          ("openai",     "gpt-4o"),
    "Claude Haiku (Anthropic)": ("anthropic",  "claude-haiku-4-5-20251001"),
    "Mistral 7B (OpenRouter)":  ("openrouter", "mistralai/mistral-7b-instruct"),
}


def build_app():
    import gradio as gr

    def load_sample(name):
        if name in SAMPLE_PAIRS:
            return SAMPLE_PAIRS[name]
        return "", ""

    def scan(source, claim, model_name, run_llm):
        if not source.strip() or not claim.strip():
            return "⚠️ Please enter both a source text and a claim.", "", "", ""

        # Rule-based
        signals = rule_scan(source, claim)
        rule_verdict = "🔴 LIKELY HALLUCINATION" if signals else "🟢 LIKELY CLEAN"
        rule_output  = f"**Rule-based verdict: {rule_verdict}**\n\n"
        if signals:
            rule_output += "Signals detected:\n" + "\n".join(f"• {s}" for s in signals)
        else:
            rule_output += "No rule violations detected."

        # LLM judge
        llm_output = latency_text = ""
        if run_llm:
            if model_name not in MODEL_CONFIG:
                llm_output = "Unknown model selected."
            else:
                provider, model_id = MODEL_CONFIG[model_name]
                key_map = {
                    "openai":     "OPENAI_API_KEY",
                    "anthropic":  "ANTHROPIC_API_KEY",
                    "openrouter": "OPENROUTER_API_KEY",
                }
                key = os.environ.get(key_map.get(provider, ""), "")
                if not key:
                    llm_output = f"❌ {key_map[provider]} not set."
                else:
                    try:
                        if provider == "anthropic":
                            import anthropic
                            client = anthropic.Anthropic(api_key=key)
                        else:
                            from openai import OpenAI
                            kwargs = {"api_key": key}
                            if provider == "openrouter":
                                kwargs["base_url"] = "https://openrouter.ai/api/v1"
                            client = OpenAI(**kwargs)

                        result, latency = llm_scan(source, claim, model_name, provider, model_id, client)
                        llm_output   = f"**LLM Judge ({model_name}):**\n\n{result}"
                        latency_text = f"⏱ {latency:.2f}s"
                    except Exception as e:
                        llm_output = f"❌ Error: {e}"

        # Similarity
        from detectors.entailment import _cosine_similarity
        sim = _cosine_similarity(source, claim)
        sim_text = f"Semantic similarity: **{sim:.3f}** ({'high' if sim > 0.5 else 'low' if sim < 0.25 else 'moderate'})"

        return rule_output, llm_output, sim_text, latency_text

    with gr.Blocks(title="Hallucination Detector") as demo:
        gr.Markdown("# 🔍 Hallucination Detection & Mitigation\n**P8 · prompt-engineering-lab** — Scan claims for faithfulness to source")

        with gr.Row():
            sample_dd = gr.Dropdown(
                choices=list(SAMPLE_PAIRS.keys()),
                label="Load a sample pair",
                value=None,
            )
            load_btn = gr.Button("Load →", size="sm")

        with gr.Row():
            source_box = gr.Textbox(label="Source Text (ground truth)", lines=5,
                                    placeholder="Paste the source/reference text here...")
            claim_box  = gr.Textbox(label="Claim to Check", lines=5,
                                    placeholder="Paste the claim to evaluate...")

        with gr.Row():
            model_dd  = gr.Dropdown(choices=list(MODEL_CONFIG.keys()),
                                    value="GPT-4o-mini (OpenAI)", label="LLM Judge Model")
            run_llm   = gr.Checkbox(label="Enable LLM Judge (costs API tokens)", value=False)

        scan_btn = gr.Button("🔍 Scan for Hallucinations", variant="primary")

        rule_out    = gr.Markdown(label="Rule-Based Detection")
        llm_out     = gr.Markdown(label="LLM Judge")
        sim_out     = gr.Markdown()
        latency_out = gr.Markdown()

        gr.Markdown("""---
**Detection methods:**
- **Rule-based** — Free, instant. Catches numeric errors, entity invention, unsupported superlatives
- **LLM Judge** — More accurate, catches semantic hallucinations. Requires API key
- **Semantic similarity** — Cosine distance between source and claim
        """)

        load_btn.click(fn=load_sample, inputs=sample_dd, outputs=[source_box, claim_box])
        sample_dd.change(fn=load_sample, inputs=sample_dd, outputs=[source_box, claim_box])
        scan_btn.click(
            fn=scan,
            inputs=[source_box, claim_box, model_dd, run_llm],
            outputs=[rule_out, llm_out, sim_out, latency_out],
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
