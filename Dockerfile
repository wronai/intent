# IntentForge - Production Dockerfile
# Multi-stage build for minimal image size

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project files
COPY pyproject.toml .

# Install Python dependencies with cache mount
# We install all dependencies here including those previously in runtime
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install ".[server]" && \
    pip install \
    pytesseract pillow \
    requests httpx aiohttp \
    numpy pandas \
    beautifulsoup4 lxml \
    pyyaml toml \
    pdf2image

# =============================================================================
# Stage 2: Tester
# =============================================================================
FROM builder AS tester

# Install unit/integration test dependencies (no Playwright)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ".[test]"

# Copy source code for testing
COPY intentforge/ ./intentforge/
COPY examples/ ./examples/
COPY tests/ ./tests/
COPY pytest.ini .

# Run tests with timeout
# Fails build if tests fail
RUN pytest --timeout=300 --ignore=tests/e2e -m "not e2e" tests/

# =============================================================================
# Stage 3: E2E (Playwright)
# =============================================================================
FROM builder AS e2e

# Install Playwright (and only e2e extras) + browser binaries
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ".[test,e2e]" && \
    playwright install --with-deps chromium

# Copy source code for testing
COPY intentforge/ ./intentforge/
COPY examples/ ./examples/
COPY tests/ ./tests/
COPY pytest.ini .

# Run only e2e tests
RUN pytest --timeout=300 -m e2e tests/e2e/

# =============================================================================
# Stage 4: Runtime
# =============================================================================
FROM python:3.12-slim AS runtime

# Labels
LABEL maintainer="Softreck <info@softreck.dev>"
LABEL version="0.1.0"
LABEL description="IntentForge - NLP-driven Code Generation Framework"

# Install Tesseract OCR and language packs (System dependencies)
# Combine apt-get calls to reduce layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-pol \
    tesseract-ocr-eng \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r intentforge && useradd -r -g intentforge intentforge

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Make venv writable (optional, usually not needed in immutable containers but kept for flexibility)
RUN chmod -R a+w /opt/venv

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
CMD ["sh", "-c", "python examples/server.py"]

# =============================================================================
# Alternative targets
# =============================================================================

# Worker target
FROM runtime AS worker
CMD ["python", "-m", "intentforge.worker"]

# CLI target
FROM runtime AS cli
ENTRYPOINT ["python", "-m", "intentforge.cli"]
