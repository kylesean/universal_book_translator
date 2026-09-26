# UBT OCR Sidecar Container (Zero dependency pollution on UBT host)
# Pin the base tag; pin the whole layer by digest when your registry mirror is
# fixed (FROM python:3.11-slim-bookworm@sha256:...).
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system dependencies for OpenCV and imaging
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pin dependency versions so image rebuilds are reproducible.
RUN pip install --no-cache-dir \
    fastapi==0.115.6 \
    uvicorn==0.34.0 \
    python-multipart==0.0.20 \
    pillow==11.1.0 \
    numpy==2.2.1 \
    rapidocr-onnxruntime==1.4.4

COPY server.py /app/server.py

# Drop root: OCR/decoding is untrusted input processing.
RUN useradd --create-home --uid 10001 ubt && chown -R ubt:ubt /app
USER ubt

EXPOSE 8765

HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
  CMD curl -fsS -H "Authorization: Bearer ${UBT_OCR_SIDECAR_TOKEN}" http://localhost:8765/health || exit 1

# UBT_OCR_SIDECAR_TOKEN enables the bearer gate; set it before exposing the port.
# Run with resource limits, e.g.:
#   docker run --memory 2g --pids-limit 256 -e UBT_OCR_SIDECAR_TOKEN=... -p 8765:8765 ...
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8765"]
