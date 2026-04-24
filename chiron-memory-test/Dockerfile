# syntax=docker/dockerfile:1
# Chiron Memory Schema Test — RunPod Serverless container
#
# Build: docker build -t chiron-memory-test .
# The container loads gpt-oss-20b once at startup and stays warm for all
# subsequent RunPod invocations.

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

LABEL maintainer="devthedevoloper"
LABEL description="Chiron Memory Smart Chunking Schema test backend for RunPod Serverless"

# ── System dependencies ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    pip3 install --upgrade pip setuptools wheel

# ── Working directory ───────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ─────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# vLLM is required — no fallback
RUN pip3 install --no-cache-dir "vllm>=0.4.3"

# ── Application code ────────────────────────────────────────────────────────
COPY . .

# ── Model cache volume mount point ──────────────────────────────────────────
# On RunPod, attach a Network Volume at /runpod-volume and set
# MODEL_CACHE_DIR=/runpod-volume/model-cache to persist weights across restarts.
RUN mkdir -p /runpod-volume/model-cache

# ── Environment defaults (override via RunPod dashboard) ────────────────────
ENV MODEL_NAME=openai/gpt-oss-20b
ENV MODEL_CACHE_DIR=/runpod-volume/model-cache
ENV HF_TOKEN=""
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Entry point ─────────────────────────────────────────────────────────────
# handler.py loads the model at import time, then starts the RunPod listener.
CMD ["python3", "handler.py"]
