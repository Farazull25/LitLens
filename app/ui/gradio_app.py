"""Multi-tab Gradio UI mounted inside FastAPI on Hugging Face Spaces (CPU Basic)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
from eval.benchmark_grid import run_benchmark_grid

from app.agent.tools.build_lit_matrix import run_build_lit_matrix
from app.agent.tools.fetch_parse_pdf import run_fetch_and_parse_pdf, run_ingest_uploaded_pdf
from app.agent.tools.format_citations import run_format_citations
from app.api.routes import get_copilot_agent
from app.observability.metrics import metrics_registry
from app.ui.components import format_chunks_inspector_markdown


def _handle_chat(
    message: str,
    history: list[list[str]],
    filter_ids_csv: str,
) -> tuple[str, list[list[str]], str]:
    agent = get_copilot_agent()
    filter_ids = [x.strip() for x in filter_ids_csv.split(",") if x.strip()] if filter_ids_csv else None
    try:
        state = agent.run(query=message, arxiv_ids=filter_ids)
    except ValueError as exc:
        err_msg = f"**Guardrail Rejection**: {exc}"
        return "", [*history, [message, err_msg]], "### Security Alert\nRequest blocked by input validation."

    inspector_md = format_chunks_inspector_markdown(
        chunks=state.get("retrieved_chunks", []),
        citations=state.get("citations", []),
        max_confidence=float(state.get("max_confidence", 0.0)),
        cached_hit=bool(state.get("cached_hit", False)),
    )
    answer = state.get("final_answer", "")
    return "", [*history, [message, answer]], inspector_md


def _handle_pdf_uploads(file_paths: list[str] | None) -> tuple[str, list[list[Any]]]:
    agent = get_copilot_agent()
    if not file_paths:
        papers = agent.vector_store.list_papers()
        return "Please select one or more PDF files to upload.", [
            [p["arxiv_id"], p["title"], p["authors"], p["published"]] for p in papers
        ]

    messages: list[str] = []
    for fp in file_paths:
        path_obj = Path(fp)
        try:
            pdf_bytes = path_obj.read_bytes()
            res = run_ingest_uploaded_pdf(
                pdf_bytes=pdf_bytes,
                filename=path_obj.name,
                vector_store=agent.vector_store,
            )
            messages.append(
                f"- Uploaded & Indexed **{res['title']}** (`{res['arxiv_id']}`): `{res['chunks_total']}` chunks."
            )
        except Exception as exc:
            messages.append(f"- Failed `{path_obj.name}`: {exc}")

    papers = agent.vector_store.list_papers()
    rows = [[p["arxiv_id"], p["title"], p["authors"], p["published"]] for p in papers]
    return "\n".join(messages), rows


def _handle_ingest(arxiv_id: str) -> tuple[str, list[list[Any]]]:
    agent = get_copilot_agent()
    try:
        res = run_fetch_and_parse_pdf({"arxiv_id": arxiv_id}, vector_store=agent.vector_store)
        msg = (
            f"Ingested **{res['title']}** (`{res['arxiv_id']}`) — "
            f"`{res['chunks_total']}` chunks indexed (Source: `{res['source']}`)."
        )
    except Exception as exc:
        msg = f"Error ingesting `{arxiv_id}`: {exc}"

    papers = agent.vector_store.list_papers()
    rows = [[p["arxiv_id"], p["title"], p["authors"], p["published"]] for p in papers]
    return msg, rows


def _handle_matrix(arxiv_ids_csv: str, focus_topic: str) -> tuple[str, str]:
    agent = get_copilot_agent()
    ids = [x.strip() for x in arxiv_ids_csv.split(",") if x.strip()]
    res = run_build_lit_matrix(
        {"arxiv_ids": ids, "topic": focus_topic or "methodology and results"},
        vector_store=agent.vector_store,
        llm=agent.llm,
    )
    cites = run_format_citations({"arxiv_ids": ids, "style": "both"}, vector_store=agent.vector_store)
    bib_md = f"### APA References\n{cites['apa']}\n\n### BibTeX\n```bibtex\n{cites['bibtex']}\n```"
    return res["markdown"], bib_md


def _handle_subscribe(email: str, topic: str) -> tuple[str, list[list[Any]]]:
    agent = get_copilot_agent()
    if "@" not in email:
        return "Please enter a valid email address.", []
    agent.ddb_store.subscribe_topic(email=email, topic=topic)
    subs = agent.ddb_store.list_subscriptions()
    rows = [[s["email"], s["topic"]] for s in subs]
    return f"Subscribed `{email}` to weekly arXiv digest for **{topic}**.", rows


def _handle_eval_and_telemetry() -> tuple[str, dict[str, Any]]:
    grid = run_benchmark_grid()
    return grid["markdown"], metrics_registry.snapshot()


def create_gradio_blocks() -> gr.Blocks:
    """Construct the LitLens multi-tab Gradio Blocks interface."""
    agent = get_copilot_agent()
    initial_papers = [
        [p["arxiv_id"], p["title"], p["authors"], p["published"]]
        for p in agent.vector_store.list_papers()
    ]

    with gr.Blocks(title="LitLens — Research Paper Copilot") as demo:
        gr.Markdown(
            "# 🔬 LitLens: Production Research Paper Copilot\n"
            "**100% Free-Tier Stack**: LangGraph Plan-then-Execute • Local Qwen2.5 GGUF + `bge-small` • "
            "Hybrid FAISS + BM25 • AWS Free-Tier (S3, DynamoDB Provisioned 5/5, Lambda Function URLs, EventBridge + SES)"
        )

        with gr.Tabs():
            with gr.Tab("1. Copilot Chat & PDF Upload"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="LitLens Citation-Grounded Chat",
                            height=420,
                        )
                        q_input = gr.Textbox(
                            label="Ask a question about uploaded PDFs or indexed arXiv papers",
                            placeholder="e.g., What methodology and evaluation metrics are used in the uploaded paper?",
                            lines=2,
                        )
                        with gr.Row():
                            filter_input = gr.Textbox(
                                label="Optional Paper ID Filter (comma-separated)",
                                placeholder="e.g., 2401.10001, upload-mypaper-a1b2c3",
                            )
                            ask_btn = gr.Button("Send Message", variant="primary")
                            clear_btn = gr.Button("Clear Chat")
                    with gr.Column(scale=2):
                        pdf_upload = gr.File(
                            label="Upload Your Research Paper PDFs (Sanitized & Indexed Locally)",
                            file_count="multiple",
                            file_types=[".pdf"],
                            type="filepath",
                        )
                        upload_btn = gr.Button("Parse, Sanitize & Index Uploaded PDF(s)", variant="secondary")
                        upload_status = gr.Markdown()
                        inspector_md = gr.Markdown(
                            value="*Retrieved chunks and `[paper_id#cN]` verification badges appear here.*"
                        )

            with gr.Tab("2. Ingest by arXiv ID & Corpus Table"):
                with gr.Row():
                    ingest_id_box = gr.Textbox(
                        label="arXiv Paper ID",
                        placeholder="e.g., 2005.11401 or 2312.10997",
                    )
                    ingest_btn = gr.Button("Fetch, Sanitize & Index arXiv PDF", variant="primary")
                ingest_status = gr.Markdown()
                papers_table = gr.Dataframe(
                    headers=["Paper / arXiv ID", "Title", "Authors", "Published"],
                    value=initial_papers,
                    label="Indexed Papers (Uploaded PDFs + arXiv Corpus)",
                )
                ingest_btn.click(
                    _handle_ingest,
                    inputs=[ingest_id_box],
                    outputs=[ingest_status, papers_table],
                )
                upload_btn.click(
                    _handle_pdf_uploads,
                    inputs=[pdf_upload],
                    outputs=[upload_status, papers_table],
                )
                ask_btn.click(
                    _handle_chat,
                    inputs=[q_input, chatbot, filter_input],
                    outputs=[q_input, chatbot, inspector_md],
                )
                q_input.submit(
                    _handle_chat,
                    inputs=[q_input, chatbot, filter_input],
                    outputs=[q_input, chatbot, inspector_md],
                )
                clear_btn.click(lambda: ([], ""), inputs=[], outputs=[chatbot, inspector_md])

            with gr.Tab("3. Literature Review Matrix"):
                with gr.Row():
                    matrix_ids = gr.Textbox(
                        label="arXiv IDs to Compare (leave blank to compare top indexed papers)",
                        placeholder="2401.10001, 2401.10002, 2401.10003, 2401.10004",
                    )
                    matrix_topic = gr.Textbox(
                        label="Comparison Dimension Focus",
                        value="problem, methodology, benchmarks, and quantitative findings",
                    )
                matrix_btn = gr.Button("Generate Map-Reduce Comparison Matrix", variant="primary")
                matrix_output = gr.Markdown()
                citations_output = gr.Markdown()
                matrix_btn.click(
                    _handle_matrix,
                    inputs=[matrix_ids, matrix_topic],
                    outputs=[matrix_output, citations_output],
                )

            with gr.Tab("4. Weekly Digest (EventBridge + SES)"):
                gr.Markdown(
                    "Subscribe to receive a weekly HTML digest of new arXiv papers every Monday at 08:00 UTC "
                    "via AWS EventBridge, Lambda, and Amazon SES."
                )
                with gr.Row():
                    sub_email = gr.Textbox(label="Verified Recipient Email", placeholder="researcher@university.edu")
                    sub_topic = gr.Textbox(label="arXiv Topic / Query", placeholder="Retrieval Augmented Generation")
                sub_btn = gr.Button("Save Subscription to DynamoDB", variant="primary")
                sub_status = gr.Markdown()
                sub_table = gr.Dataframe(
                    headers=["Email", "Subscribed Topic"],
                    label="Active Weekly Digest Subscriptions",
                )
                sub_btn.click(_handle_subscribe, inputs=[sub_email, sub_topic], outputs=[sub_status, sub_table])

            with gr.Tab("5. Evaluation Harness & Telemetry"):
                gr.Markdown(
                    "Run the 30-question Golden Evaluation Benchmark across chunk sizes (`256`, `512`, `1024`) "
                    "and model sizes (`1.5B`, `3B`), and inspect live CloudWatch-compatible metrics."
                )
                eval_btn = gr.Button("Run 30-Question Benchmark Grid", variant="primary")
                eval_md = gr.Markdown()
                telemetry_json = gr.JSON(label="Live Telemetry Counters & Latencies")
                eval_btn.click(_handle_eval_and_telemetry, inputs=[], outputs=[eval_md, telemetry_json])

    return demo
