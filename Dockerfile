# IntentForge - Production Dockerfile
# Multi-stage build for minimal image size

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir ".[server]"

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.11-slim AS runtime

# Labels
LABEL maintainer="Softreck <info@softreck.dev>"
LABEL version="0.1.0"
LABEL description="IntentForge - NLP-driven Code Generation Framework"

# Create non-root user
RUN groupadd -r intentforge && useradd -r -g intentforge intentforge

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY intentforge/ ./intentforge/
COPY examples/ ./examples/

# Create directories
RUN mkdir -p /app/generated /app/cache /app/logs \
    && chown -R intentforge:intentforge /app

# Environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MQTT_HOST=mqtt \
    MQTT_PORT=1883 \
    REDIS_URL=redis://redis:6379/0 \
    DB_HOST=postgres \
    DB_PORT=5432 \
    LOG_LEVEL=INFO

# Switch to non-root user
USER intentforge

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from intentforge import IntentForge; print('OK')" || exit 1

# Default command - run API server
EXPOSE 8000
CMD ["sh", "-c", "python -m uvicorn intentforge.server:app --host 0.0.0.0 --port ${PORT:-8000}"]

# =============================================================================
# Alternative targets
# =============================================================================

# Worker target
FROM runtime AS worker
CMD ["python", "-m", "intentforge.worker"]

# CLI target
FROM runtime AS cli
ENTRYPOINT ["python", "-m", "intentforge.cli"]
