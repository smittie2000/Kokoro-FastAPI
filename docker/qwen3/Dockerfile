# Qwen3-TTS streaming service
# Based on Kokoro-FastAPI's GPU image with Qwen3 dependencies added
FROM nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/api
ENV USE_GPU=true
ENV TTS_ENGINE=qwen3

# System dependencies (ninja-build required for flash-attn compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-dev \
    espeak-ng ffmpeg git curl sox libsox-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -g 1001 appuser && useradd -m -u 1001 -g appuser appuser

WORKDIR /app

# Create venv
RUN python3 -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install base dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[gpu]"

# Install Qwen3-TTS streaming dependencies (excluding flash-attn, installed separately)
COPY requirements-qwen3.txt ./
RUN pip install --no-cache-dir $(grep -v 'flash-attn' requirements-qwen3.txt | grep -v '^#' | grep -v '^$')

# Install flash-attn from source (separate layer — ~10min compile, cached independently)
# --no-build-isolation: uses already-installed PyTorch to find CUDA headers
# MAX_JOBS: limit parallel compilation to avoid OOM during Docker build
RUN MAX_JOBS=4 pip install --no-cache-dir --no-build-isolation flash-attn>=2.7.0

# Copy application code + vendored qwen_tts package
COPY api/ ./api/
COPY qwen_tts/ ./qwen_tts/
COPY voices/ ./voices/

# Create model cache directory
RUN mkdir -p /app/models && chown -R appuser:appuser /app

USER appuser

# Model will be downloaded on first startup if not mounted
EXPOSE 8880

CMD ["python", "-m", "uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8880"]
