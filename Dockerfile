# ── OCR Project Dockerfile (Monolith: Backend + Frontend) ──
FROM python:3.11-slim

# ── System dependencies for OpenCV ────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ──────────────────────────────────
COPY app.py .
COPY ocr_config.py .
COPY templates/ ./templates/

# ── Copy model checkpoint (REQUIRED) ───────────────────────
COPY checkpoints/ocr_predict.keras ./checkpoints/ocr_predict.keras

# ── Environment variables ──────────────────────────────────
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# ── Expose port ────────────────────────────────────────────
EXPOSE 8080

# ── Health check ───────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/')" || exit 1

# ── Run with Gunicorn (production WSGI server) ─────────────
CMD exec gunicorn \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --threads 4 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    app:app
