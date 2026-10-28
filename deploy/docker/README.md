# Universal Book Translator (UBT) Container Deployment Guide

This directory provides production-ready, reproducible container deployment configurations for Universal Book Translator.

## Highlights
- **All-in-One Out-of-the-Box**: Pre-baked with **Typst 0.15.1** compiler and **Noto CJK fonts** (`fonts-noto-cjk`), ensuring 100% tofu-free document rendering for Chinese, Japanese, and Korean books without local host setup.
- **Multi-Architecture**: Supports both `linux/amd64` (Intel/AMD) and `linux/arm64` (Apple Silicon / AWS Graviton) with pinned SHA256 integrity checks.
- **Security Hardened**: Runs as non-root user `ubt` (UID 10001) under least-privilege principles.
- **Zero-Friction Local Deployment**: Built-in Web Console, REST API, CLI, Queue Worker, and MCP server options.

---

## 1. Quick Start with Docker Compose

1. Clone the repository and navigate to the project root:
   ```bash
   git clone https://github.com/your-org/universal_book_translator.git
   cd universal_book_translator
   ```

2. Start the UBT service:
   ```bash
   docker compose up -d
   ```

3. Open the Web Console in your browser:
   ```
   http://localhost:8000
   ```

4. View service health and logs:
   ```bash
   docker compose ps
   docker compose logs -f ubt
   ```

---

## 2. Running via Docker CLI Directly

### Starting the API & Web Console Server
```bash
docker run -d \
  --name ubt-app \
  -p 8000:8000 \
  -v "$(pwd)/data:/data" \
  -e UBT_PROVIDER=openai \
  -e OPENAI_API_KEY=sk-... \
  universal-book-translator:latest
```

### Running CLI Translation Commands
```bash
docker run --rm \
  -v "$(pwd)/data:/data" \
  -e OPENAI_API_KEY=sk-... \
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
