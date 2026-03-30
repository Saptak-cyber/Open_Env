# Multi-stage Dockerfile for PR Code Review Assistant

# Builder stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml ./
COPY pr_review_env/ ./pr_review_env/
COPY tasks/ ./tasks/
COPY inference.py ./

# Install dependencies with uv
RUN uv pip install --system --no-cache .

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Create non-root user FIRST
RUN useradd -m -u 1000 appuser

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY pr_review_env/ ./pr_review_env/
COPY tasks/ ./tasks/
COPY inference.py ./
COPY openenv.yaml ./
COPY pyproject.toml ./

# Fix ownership
RUN chown -R appuser:appuser /app

# Setup environment
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run server
CMD ["python", "-m", "uvicorn", "pr_review_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
