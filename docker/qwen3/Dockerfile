# Qwen3-TTS streaming service
# Based on Kokoro-FastAPI's GPU image with Qwen3 dependencies added
FROM nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/api
ENV USE_GPU=true
ENV TTS_ENGINE=qwen3

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    espeak-ng ffmpeg git curl sox libsox-dev \
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

# Install Qwen3-TTS streaming dependencies
COPY requirements-qwen3.txt ./
RUN pip install --no-cache-dir -r requirements-qwen3.txt

# Copy application code
COPY api/ ./api/
COPY voices/ ./voices/

# Create model cache directory
RUN mkdir -p /app/models && chown -R appuser:appuser /app

USER appuser

# Model will be downloaded on first startup if not mounted
EXPOSE 8880

CMD ["python", "-m", "uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8880"]
