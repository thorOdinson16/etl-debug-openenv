# ── Stage 1: dependency installation ─────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps (needed for some numpy/pandas wheels on slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL org.opencontainers.image.title="ETL Debug OpenEnv"
LABEL org.opencontainers.image.description="OpenEnv-compliant AI agent environment for ETL pipeline debugging"
LABEL org.opencontainers.image.version="1.0.0"

EXPOSE 7860

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY app/        ./app/
COPY models/     ./models/
COPY tasks/      ./tasks/
COPY api/        ./api/
COPY inference.py .
COPY openenv.yaml .

# Create non-root user and fix permissions
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

# Health check — pings the /health endpoint every 30s
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=15s \
    --retries=3 \
    CMD python -c \
        "import urllib.request, sys; \
         r = urllib.request.urlopen('http://localhost:7860/health', timeout=8); \
         sys.exit(0 if r.status == 200 else 1)"

# Environment defaults (can be overridden at runtime)
ENV MODEL_NAME="gpt-4o"
ENV ENV_BASE_URL="http://localhost:7860"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--log-level", "info"]