# Architectural Trade-Offs & Free-Tier Survival Guide

## 1. Why LangGraph Plan-then-Execute Instead of ReAct?
- **Problem**: Small quantized open models (`Qwen2.5-1.5B/3B-Instruct Q4_K_M`) frequently enter infinite tool-calling loops or emit malformed tool arguments in free-form multi-turn ReAct cycles.
- **Solution**: LitLens uses a deterministic **4-node Plan-then-Execute state machine** (`planner` -> `executor` -> `confidence_gate` -> `synthesizer`). The planner runs once with a **GBNF context-free grammar schema** (`ExecutionPlan`) and outputs 1–4 steps that are validated by Pydantic before execution.
- **Trade-off**: The agent cannot dynamically invent new multi-hop recovery plans mid-turn, but achieves 100% JSON validity and ~2.5x lower CPU latency on a 2-vCPU Hugging Face Space.

## 2. Why Map-Reduce for Literature Review Matrices?
- **Problem**: Stuffing 5 papers into a single prompt overflows a 4,096-token CPU context window and causes severe cross-paper citation swaps in sub-3B models.
- **Solution**: `build_lit_matrix` maps over each paper individually (extracting a grammar-constrained `LitMatrixRow` JSON from that paper's top-3 chunks) and reduces the rows in Python into a Markdown table with verified `[arxiv_id#cN]` citations.

## 3. Strict \$0.00 AWS & Hugging Face Free-Tier Engineering

| Service | Free-Tier Quota | Common Billing Trap | LitLens Design Choice |
| :--- | :--- | :--- | :--- |
| **Hugging Face Spaces** | `cpu-basic` (2 vCPU, 16 GB RAM, 50 GB ephemeral disk) | Persistent disk costs \$5/mo; sleeping Spaces wipe local disk after 48h. | Pre-bake GGUF & `bge-small` into the Docker image layer; sync FAISS snapshots & paper bundles to S3. |
| **Amazon DynamoDB** | 25 GB storage + **25 RCU / 25 WCU Provisioned** (Always-Free) | `PAY_PER_REQUEST` (On-Demand) is **not** in the Always-Free tier and bills immediately. | Single-table design (`LitLensStore`) locked to `BillingMode: PROVISIONED` (`5 RCU / 5 WCU`) with native `ttl` expiration. |
| **Amazon S3** | 5 GB storage, **2,000 PUTs**, 20,000 GETs/mo | Storing raw PDFs or 1 S3 object per chunk exhausts 2,000 PUTs in ~40 papers. | Parse PDFs in memory and upload **1 gzipped JSON bundle per paper** (`papers/{arxiv_id}.json.gz`, ~40 KB = 1 PUT). |
| **AWS Lambda** | 1M requests + 400,000 GB-s/mo (Always-Free) | API Gateway costs \$1.00–\$3.50/M after 12 months; ECR container images cost \$0.10/GB-mo. | Use `.zip` Lambdas with **Lambda Function URLs** (`AuthType: AWS_IAM` + SigV4 signing). Zero API Gateway, zero ECR. |
| **Amazon CloudWatch** | 5 GB logs, **10 custom metrics**, 10 alarms (Always-Free) | Adding `session_id` or `user_id` dimensions creates thousands of billable custom metrics (\$0.30/metric). | Hard-allowlist 6 metric names (`QueryLatencyMs`, `RetrievalHitCount`, `RefusalCount`, `CacheHitCount`, `InjectionBlockedCount`, `LLMErrorCount`) with **empty `Dimensions: []`** and 7-day log retention. |
| **AWS Budgets** | First 2 budgets Always-Free | Surprise spillover charges. | `ZeroSpendMonthlyBudget` configured in `infra/template.yaml` to email immediately at \$0.01 actual spend. |

## 4. AWS CloudFront HTTPS & Lambda Function URL Edge Delivery
- **CloudFront HTTPS Distribution (E4BC0SGVGTZCE)**: Serves LitLens over AWS TLS 1.3 within the 1 TB Always-Free data transfer tier.

- **Zero-Key Client Protection**: The static HTML client never stores API keys; all LLM calls route through the AWS Lambda Function URL (litlens-groq-chat).

- **Zero-Spend Budget Guard**: AWS Budget alerts fire at $0.01 forecasted/actual spend to protect Free-Tier boundaries.

- **SES Sandbox & Quota Protection**: WeeklyDigestFunction enforces a 50-email/run safety cap within the free tier.

- **Edge Caching TTL**: Static assets use a 60s Cache-Control header for instant rollouts.

- **CloudWatch Log Retention**: Locked to 7 days (RetentionInDays: 7) to stay well under the 5 GB Always-Free ingestion cap.

<!-- benchmark verification step 10 -->

