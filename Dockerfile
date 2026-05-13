# ╔══════════════════════════════════════════════════════════════════╗
# ║    IPL Win Prediction — Multi-Stage Dockerfile                  ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── Base Stage ───────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS base

LABEL maintainer="Aranya2801"
LABEL version="2.0.0"
LABEL description="IPL Win Prediction Model — Production Container"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies Stage ───────────────────────────────────────────────
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── API Stage ────────────────────────────────────────────────────────
FROM deps AS api

COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--log-level", "info"]

# ── Streamlit Stage ──────────────────────────────────────────────────
FROM deps AS streamlit

COPY streamlit_app/ ./streamlit_app/
COPY src/ ./src/
COPY models/ ./models/

RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]

# ── Training Stage (for CI/CD) ───────────────────────────────────────
FROM deps AS trainer

COPY . .

CMD ["python", "src/models/train_model.py"]
