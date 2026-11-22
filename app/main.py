"""FastAPI application entrypoint mounting the Gradio UI at `/` and REST routes under `/api`."""

from __future__ import annotations

import gradio as gr
import uvicorn
from fastapi import FastAPI

from app.api.middleware import RateLimitAndTraceMiddleware
from app.api.routes import router as api_router
from app.config import get_settings
from app.observability.logger import configure_logging
from app.ui.gradio_app import create_gradio_blocks


def create_app() -> FastAPI:
    """Create and configure the production FastAPI + Gradio application."""
    configure_logging()
    get_settings()

    fastapi_app = FastAPI(
        title="LitLens — Research Paper Copilot",
        version="1.0.0",
        description="Citation-grounded arXiv research copilot running on 100% free AWS & Hugging Face tiers.",
    )
    fastapi_app.add_middleware(RateLimitAndTraceMiddleware)
    fastapi_app.include_router(api_router)

    gradio_blocks = create_gradio_blocks()
    fastapi_app = gr.mount_gradio_app(fastapi_app, gradio_blocks, path="/")
    return fastapi_app


app = create_app()

if __name__ == "__main__":
    cfg = get_settings()
    uvicorn.run("app.main:app", host=cfg.host, port=cfg.port, reload=False)
