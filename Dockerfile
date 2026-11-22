# ─────────────────────────────────────────────────────────────────────────────
# The Value Hunter — FastAPI Production Image (api_server.py + index.html)
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=America/New_York \
    PORT=8000 \
    VH_DATA_DIR=/app/var

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application source (see .dockerignore for exclusions: .env, *.db, cache/ ...)
COPY . .

# Run as an unprivileged user; give it a writable HOME (yfinance cache) and data dir (SQLite)
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/var /app/cache \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/healthz', timeout=4)" || exit 1

# Shell form so $PORT (injected by Render / Railway / Cloud Run / Fly) is expanded.
# Single worker: SQLite + in-memory quote caches are process-local.
CMD exec uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips="*"

