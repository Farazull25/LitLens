# Model & System Card: LitLens Research Paper Copilot

## 1. System Overview
**LitLens** is a citation-grounded academic Research Paper Copilot engineered to run strictly within **\$0.00 Free-Tier boundaries** across **Hugging Face Spaces (`cpu-basic`: 2 vCPU, 16 GB RAM)** and **AWS Free Tier** (Provisioned DynamoDB, S3, Lambda Function URLs, EventBridge, SES, SSM, CloudWatch, and AWS Budgets).

## 2. Models Used

| Component | Model ID | Quantization / Format | License | Memory Footprint | Execution Environment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Local LLM (Default)** | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | `Q4_K_M` (4-bit GGUF) | Apache-2.0 | ~1.1 GB RAM | Local CPU (`llama-cpp-python`, 2 threads) |
| **Alternative Local LLM** | `Qwen/Qwen2.5-3B-Instruct-GGUF` | `Q4_K_M` (4-bit GGUF) | Apache-2.0 | ~2.1 GB RAM | Local CPU (`llama-cpp-python`, 2 threads) |
| **Optional Cloud Fallback** | `Qwen/Qwen2.5-3B-Instruct` | FP16 / BF16 Serverless | Apache-2.0 | 0 MB Local | Free Hugging Face Serverless Inference API (`hf-inference`) |
| **Dense Embeddings** | `BAAI/bge-small-en-v1.5` | 384-dim FP32 / ONNX | MIT | ~130 MB RAM | Local CPU (`sentence-transformers`) |
| **Sparse Lexical Search** | `BM25Okapi` (`rank-bm25`) | Inverted Token Index | Apache-2.0 | <15 MB RAM | Local CPU |

## 3. Intended Use Cases
1. **Verifiable Academic Q&A**: Answering technical research questions over ingested arXiv papers where every factual claim is tied to an explicit `[arxiv_id#cN]` chunk identifier.
2. **Literature Review Comparison Matrix Generation**: Comparing up to 5 papers across *Problem*, *Methodology*, *Datasets/Benchmarks*, *Key Findings*, and *Limitations* using a Map-Reduce extraction pipeline.
3. **Automated Weekly Topic Digest**: Delivering a scheduled Monday morning HTML digest of newly submitted arXiv papers on user-followed topics via AWS EventBridge, Lambda, and Amazon SES.

## 4. Safety, Grounding & Prompt-Injection Defenses
- **Untrusted PDF Treatment**: Academic PDFs from open repositories can contain hidden or adversarial text. LitLens strips microscopic font spans (`< 2.0pt`), white-on-white text (`0xFFFFFF`), and zero-width characters during `PyMuPDF` parsing.
- **XML Context Sandboxing**: All retrieved chunks are wrapped in `<untrusted_paper_content chunk_id="...">` tags with explicit system instructions never to execute commands inside data blocks.
- **Deterministic Citation Verification**: A post-generation regex validator (`verify_and_clean_citations`) checks every `[arxiv_id#cN]` against the actual retrieved chunk set, stripping fabricated citations and unauthorized external links.
- **Calibrated Abstention Gate**: Queries whose top hybrid retrieval score (55% dense cosine + 35% normalized BM25 + 10% Reciprocal Rank Fusion) falls below $\tau = 0.32$ are deterministically refused rather than answered from parametric memory.

## 5. Known Limitations
- **Small Model Reasoning Ceiling**: 1.5Bâ€“3B models cannot perform deep multi-hop mathematical proofs across 10+ papers simultaneously; LitLens mitigates this via single-shot grammar-constrained planning and per-paper Map-Reduce decomposition.
- **Rasterized Figures**: Text embedded exclusively inside rasterized images/plots inside PDFs is not extracted without OCR.

## 6. Cloud LPU Acceleration Tier
- **AWS Lambda Function URL + Groq LPU (llam-2-7b / qwen/qwen3.8-27b)**: Provides sub-second general assistant and RAG synthesis over HTTPS.

- **Hybrid Confidence Calibration**: Combines 55% dense cosine similarity, 35% normalized BM25, and 10% RRF agreement.

- **S3 Snapshot Hydration**: Hydrates 10,000 FAISS chunks in <1.8s on container startup.

- **Zero-Width Character Filtering**: Strips Unicode zero-width joiners (\u200b–\u200d, \ufeff) before tokenization.

<!-- verified multi-user PR 2 -->

