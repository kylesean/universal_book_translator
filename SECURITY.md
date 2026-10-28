# Security Policy

## Supported Versions

Universal Book Translator (UBT) provides security fixes and patches for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

---

## Reporting a Vulnerability

We take the security and integrity of Universal Book Translator seriously. If you discover a potential security vulnerability, **please do not disclose it publicly or open a public GitHub issue.**

### Private Reporting Channel

Please report security issues privately via:
- **GitHub Private Vulnerability Reporting**: [Open a Private Advisory](https://github.com/universal-book-translator/universal-book-translator/security/advisories/new)
- **Email**: Send details, proof-of-concept, and environment details to `security@universal-book-translator.org` (or the repository maintainers).

### What to Include
1. Type of issue (e.g. command injection, credential leak, path traversal, XML entity expansion).
2. Clear step-by-step instructions or reproducible proof of concept.
3. Affected versions, deployment modes (Docker, CLI, API server), or environment variables.
4. Any proposed patches or mitigations.

### Response Timeline
- **Initial Response**: Within 48 hours acknowledging receipt of report.
- **Triage & Assessment**: Within 5 business days with preliminary findings.
- **Remediation & Patch**: Targeted release within 14 days depending on severity.

---

## Security Architecture & Threat Model

Universal Book Translator is designed with a defense-in-depth approach:

### 1. Zero Credential Persistence
- LLM API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`) and OCR tokens are loaded exclusively into memory from environment variables or secure storage.
- Credentials are **never** logged to disk, committed to git, or stored in SQLite job ledgers (`ubt.db`).

### 2. Untrusted Document Parsing Protection
- **XML Entity Expansion (`defusedxml`)**: All XML-based document formats (EPUB, DOCX, XLIFF, TMX) are processed through `defusedxml` to completely prevent "Billion Laughs" entity expansion, DTD external entity retrieval, and XML quadratic blowup attacks.
- **ZIP Bomb Defense**: DOCX and EPUB container extractors enforce strict uncompressed byte budgets and decompression ratio ceilings.

### 3. Subprocess & Compiler Isolation
- External formatting tools (e.g. `typst`) are executed with explicit timeouts, sandboxed temporary build directories, and pinned deterministic timestamps (`SOURCE_DATE_EPOCH`).
- Binaries downloaded in container images are pinned against immutable cryptographic SHA-256 hashes.

### 4. Container Hardening
- Official Docker images run as an unprivileged non-root user `ubt` (UID 10001) under least-privilege principles.
- The web console and API server enforce host and allowed-directory scoping (`UBT_ALLOWED_DIRS`) to prevent arbitrary filesystem traversal.
