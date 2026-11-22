# LitLens Dockerfile for Hugging Face Spaces (CPU Basic: 2 vCPU, 16 GB RAM, Free Tier)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOST=0.0.0.0 \
    MODELS_DIR=/home/user/app/models \
    LOCAL_INDEX_DIR=/home/user/app/data/faiss_index \
    HF_HOME=/home/user/app/.cache/huggingface \
    LLM_N_THREADS=2 \
    LOCAL_MODEL_SIZE=1.5B

# Install build essentials for llama-cpp-python CPU compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user required by Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app
ENV PATH="/home/user/.local/bin:${PATH}"

COPY --chown=user:user requirements.txt pyproject.toml ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir llama-cpp-python

# Pre-download free public BAAI/bge-small-en-v1.5 embeddings & Qwen2.5-1.5B-Instruct Q4_K_M GGUF
# during Docker build so HF Space cold-starts/wake-ups finish in <10 seconds.
RUN mkdir -p /home/user/app/models /home/user/app/data/faiss_index && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5'); from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Qwen/Qwen2.5-1.5B-Instruct-GGUF', filename='qwen2.5-1.5b-instruct-q4_k_m.gguf', local_dir='/home/user/app/models')"

COPY --chown=user:user . .

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
