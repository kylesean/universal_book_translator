# Universal Book Translator (UBT) Container Deployment Guide

This directory provides production-ready, reproducible container deployment configurations for Universal Book Translator.

## Highlights
- **All-in-One Out-of-the-Box**: Pre-baked with **Typst 0.15.1** compiler and **Noto CJK fonts** (`fonts-noto-cjk`), ensuring 100% tofu-free document rendering for Chinese, Japanese, and Korean books without local host setup.
- **Multi-Architecture**: Supports both `linux/amd64` (Intel/AMD) and `linux/arm64` (Apple Silicon / AWS Graviton) with pinned SHA256 integrity checks.
- **Security Hardened**: Runs as non-root user `ubt` (UID 10001) under least-privilege principles, and **fails closed**: the API will not start without `UBT_API_KEY`.
- **Zero-Friction Local Deployment**: Built-in Web Console, REST API, CLI, Queue Worker, and MCP server options.

---

## 1. Quick Start with Docker Compose

1. Clone the repository and navigate to the project root:
   ```bash
   git clone https://github.com/your-org/universal_book_translator.git
   cd universal_book_translator
   ```

2. Set an API key. **The console refuses to start without one** — it can read
   any parseable document under `/data` and spend your LLM key:
   ```bash
   echo "UBT_API_KEY=$(openssl rand -hex 24)" > .env
   ```

3. Start the UBT service. The host port is published on `127.0.0.1` by default;
   set `UBT_BIND_ADDR=0.0.0.0` only if you also keep the API key set:
   ```bash
   docker compose up -d
   ```

4. Open the Web Console in your browser (paste the key when prompted):
   ```
   http://localhost:8000
   ```

5. View service health and logs:
   ```bash
   docker compose ps
   docker compose logs -f ubt
   ```

> **Keyless local instance (development only)**: `UBT_ALLOW_NO_AUTH=1` starts the
> console without a key. Keep the port mapping on `127.0.0.1` — on a shared
> network anyone could read `/data` and burn your `UBT_LLM_API_KEY`.

---

## 2. Running via Docker CLI Directly

### Starting the API & Web Console Server
```bash
docker run -d \
  --name ubt-app \
  -p 127.0.0.1:8000:8000 \
  -v "$(pwd)/data:/data" \
  -e UBT_API_KEY="$(openssl rand -hex 24)" \
  -e UBT_PROVIDER=openai-chat \
  -e UBT_LLM_API_KEY=sk-... \
  -e UBT_DRAFT_MODEL=gpt-4o-mini \
  universal-book-translator:latest
```

### Running CLI Translation Commands
```bash
docker run --rm \
  -v "$(pwd)/data:/data" \
  -e UBT_PROVIDER=openai-chat \
  -e UBT_LLM_API_KEY=sk-... \
  -e UBT_DRAFT_MODEL=gpt-4o-mini \
  universal-book-translator:latest \
  translate /data/input/sample.pdf -o /data/output/sample_zh.pdf -l zh
```

### Running System Diagnostics (Doctor)
```bash
docker run --rm universal-book-translator:latest doctor
```

---

## 3. Storage Directory Layout

When mounting `./data:/data`:
- `/data/dbs`: SQLite job ledgers and state tracking databases.
- `/data/output`: Translated books, reports, sidecars, and companions (XLIFF, HTML, EPUB).
- `/data/input`: Source documents to translate.
- `/data/models`: Offline cache for HuggingFace/ONNX OCR models.
