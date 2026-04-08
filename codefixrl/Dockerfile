# ============================================================
# CodeFixRL Dockerfile
# Optimised for 2 vCPU / 8 GB RAM (Hugging Face Spaces)
# ============================================================
FROM python:3.11-slim

# Metadata
LABEL maintainer="CodeFixRL Authors"
LABEL description="OpenEnv-compatible environment for AI code debugging"

# System deps (minimal: only what pip needs to build wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd --create-home appuser
WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Ensure the app directory is on PYTHONPATH so local imports resolve
ENV PYTHONPATH=/app

# Runtime environment variables (override via docker run -e or HF Spaces secrets)
ENV API_BASE_URL=https://api.openai.com/v1
ENV MODEL_NAME=gpt-4o-mini
ENV TASK=easy

# Switch to non-root user
USER appuser

# Default entrypoint: run the baseline inference agent
CMD ["python", "inference.py"]
