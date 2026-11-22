---
title: LitLens Research Paper Copilot
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
hardware: cpu-basic
license: apache-2.0
short_description: Citation-grounded arXiv Copilot on 100% free HF & AWS tiers
---

# 🔬 LitLens: Production Research Paper Copilot

**LitLens** is a production-grade Research Paper Copilot agent that ingests arXiv papers, answers research questions with verifiable `[arxiv_id#cN]` chunk citations, compiles Map-Reduce literature review comparison matrices, and emails a weekly digest of new papers on topics you follow.

It is engineered from the ground up to run at **\$0.00/month** strictly within the **Hugging Face Spaces Free Tier (`cpu-basic`: 2 vCPU, 16 GB RAM)** and **AWS Free Tier** (Lambda Function URLs, S3, Provisioned DynamoDB, EventBridge, SES, CloudWatch, SSM Parameter Store, and AWS Budgets).

---

## Architecture Diagram

```mermaid
flowchart TB
    subgraph Users["User Layer"]
        U["Researcher"]
    end

    subgraph HFSpace["Hugging Face Space (cpu-basic: 2 vCPU, 16 GB RAM - Docker)"]
        UI["Gradio UI + FastAPI Backend (/api/*)"]
        GuardIn["Input Guardrails & Sliding-Window Rate Limiter"]

        subgraph LangGraph["LangGraph: Plan-then-Execute State Machine"]
            Planner["1. Constrained Planner (GBNF JSON Schema)"]
            Executor["2. Deterministic Tool Executor (Pydantic Validated)"]
            Gate["3. Calibrated Confidence Gate (Hybrid >= 0.32)"]
            Synth["4. Sandboxed XML Synthesizer & Citation Verifier"]
        end

        subgraph LocalCompute["100% Free Local CPU Inference & Hybrid RAG"]
            LLM["llama-cpp-python\n(Qwen2.5-1.5B/3B-Instruct Q4_K_M GGUF)"]
            Embed["BAAI/bge-small-en-v1.5 (384-dim)"]
            FAISS["FAISS IndexFlatIP + BM25 + RRF Fusion"]
        end

        HFFallback["HF Serverless Inference API\n(Optional Free Fallback + Circuit Breaker)"]
    end

    subgraph AWS["AWS Free-Tier Stack (AWS SAM - infra/template.yaml)"]
        DDB[("DynamoDB Single-Table (LitLensStore)\nProvisioned: 5 RCU / 5 WCU + TTL\nSessions | Memory | 24h Cache | Subscriptions | Evals")]
        S3[("Amazon S3 (< 5 GB, < 2,000 PUTs)\npapers/{id}.json.gz | FAISS Snapshots")]
        LambdaTool["Lambda: litlens-arxiv-tool\n(Lambda Function URL + AWS_IAM SigV4)"]
        EB["Amazon EventBridge\n(cron: Monday 08:00 UTC)"]
        LambdaDigest["Lambda: litlens-weekly-digest"]
        SES["Amazon SES\n(Weekly HTML Digest)"]
        CW["CloudWatch Logs (7d), 6 Zero-Dim Metrics,\n3 Alarms + $0.01 Zero-Spend AWS Budget"]
        SSM["SSM Parameter Store\n(Free Standard Tier)"]
    end

    subgraph External["External Open Repositories"]
        ArXiv["arXiv Atom API & PDF Server"]
    end

    U -->|HTTPS| UI --> GuardIn
    GuardIn <-->|24h Query Cache| DDB
    GuardIn --> Planner
    Planner <-->|GBNF JSON Plan| LLM
    LLM -.->|Fallback on Error| HFFallback
    Planner --> Executor

    Executor -->|SigV4 Signed POST| LambdaTool --> ArXiv
    LambdaTool -->|1 Gzip Bundle/Paper| S3
    Executor -->|Embed & Hybrid RRF| Embed --> FAISS
    FAISS <-->|Cold-Start Hydration| S3
    Executor -->|User Memory & Topics| DDB

    Executor --> Gate
    Gate -->|Score < 0.32: Abstain| UI
    Gate -->|Score >= 0.32: Verified Chunks| Synth
    Synth <-->|<untrusted_paper_content> XML| LLM
    Synth -->|Verified [arxiv_id#cN] Response| UI

    EB -->|Weekly Trigger| LambdaDigest
    LambdaDigest -->|Read Subscriptions| DDB
    LambdaDigest -->|Query New Papers| ArXiv
    LambdaDigest -->|Send HTML Digest| SES --> U

    HFSpace -->|JSON Logs & 6 Custom Metrics| CW
    HFSpace -->|Load Config| SSM
```

---

## Repository Structure

```text
LitLens/
├── .github/workflows/
│   ├── ci.yml                      # Ruff lint, Pytest (unit + integration + moto), 30-Q Eval Grid
│   ├── deploy_aws.yml              # AWS SAM validate, build, and OIDC deploy
│   └── deploy_hf.yml               # Automated git sync to Hugging Face Space
├── app/
│   ├── main.py                     # FastAPI entrypoint mounting Gradio UI at /
│   ├── config.py                   # Pydantic BaseSettings + AWS SSM Parameter Store loader
│   ├── api/                        # FastAPI routes (/api/chat, /ingest, /matrix, /subscribe, /health) & middleware
│   ├── ui/                         # Multi-tab Gradio Blocks UI & chunk citation inspector
│   ├── llm/                        # Provider-agnostic LLMProtocol, LlamaCpp GGUF runner, HF Fallback
│   ├── rag/                        # Section-aware chunker, bge-small embedder, FAISS + BM25 RRF reranker
│   ├── agent/                      # LangGraph Plan-then-Execute graph + 6 Pydantic-validated tools
│   ├── guardrails/                 # PDF prompt-injection sanitizer, citation verifier, rate limiter
│   ├── storage/                    # Single-table Provisioned DynamoDB & S3 compressed bundle store
│   └── observability/              # structlog JSON tracer + 6-metric CloudWatch publisher
├── infra/
│   ├── template.yaml               # Reproducible AWS SAM Free-Tier template
│   ├── samconfig.toml              # SAM deployment configuration
│   └── lambdas/                    # arxiv_tool (Function URL) & weekly_digest (EventBridge + SES)
├── eval/
│   ├── golden_dataset.json         # 30 curated Q&A pairs (including adversarial unanswerable probes)
│   ├── metrics.py                  # HitRate@K, MRR, Citation Precision/Recall, Faithfulness
│   ├── runner.py                   # End-to-end evaluation harness
│   └── benchmark_grid.py           # Grid comparison: chunk sizes (256, 512, 1024) x models (1.5B, 3B)
├── tests/                          # Unit & integration tests with moto AWS mocks
├── docs/
│   ├── MODEL_CARD.md               # Model & System Card
│   └── TRADE_OFFS.md               # Design trade-offs & Free-Tier survival guide
├── Dockerfile                      # Hugging Face Space Docker image (pre-bakes GGUF & bge-small)
└── pyproject.toml                  # Dependencies, Ruff, Mypy, and Pytest config
```

---

## Quickstart & Setup Guide

### 1. Local Development & Testing
```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# 2. Install dependencies
.venv/bin/pip install -r requirements.txt

# 3. Run Linter, Test Suite, and 30-Question Benchmark Grid
ruff check app eval infra tests
pytest -v
python -m eval.benchmark_grid

# 4. Launch FastAPI + Gradio UI locally on http://localhost:7860
python -m app.main
```

### 2. Deploying the AWS Free-Tier Stack (SAM)
```bash
cd infra
sam validate
sam build
sam deploy --guided \
  --parameter-overrides \
    EnvironmentName=prod \
    SesSenderEmail=your-verified-sender@example.com \
    AlertRecipientEmail=your-verified-recipient@example.com
```
> **Note on Amazon SES Sandbox**: In a new AWS account, verify both `SesSenderEmail` and `AlertRecipientEmail` in the Amazon SES console (`us-east-1`) so the weekly digest can send emails without leaving the free sandbox.

### 3. Deploying to Hugging Face Spaces (`cpu-basic`)
1. Create a new **Docker** Space on [Hugging Face Spaces](https://huggingface.co/new-space) with **CPU Basic (2 vCPU, 16 GB RAM — Free)**.
2. Add your repository secrets (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION=us-east-1`, `AWS_ENABLED=true`, and optional `HF_TOKEN`) in the Space settings.
3. Push the repo (or let `.github/workflows/deploy_hf.yml` push automatically on `main`).

---

## Further Documentation
- [Model & System Card](docs/MODEL_CARD.md)
- [Design Trade-Offs & Free-Tier Cost Notes](docs/TRADE_OFFS.md)
