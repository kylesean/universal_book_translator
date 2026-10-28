# syntax=docker/dockerfile:1
# Universal Book Translator (UBT) - All-in-One Production Container
# Multi-stage build with pinned Typst 0.15.1, Noto CJK fonts, and non-root execution.

# ==============================================================================
# Stage 1: Builder
# ==============================================================================
FROM python:3.12-slim-bookworm AS builder

ARG TARGETARCH
ARG TYPST_VERSION="0.15.1"
ARG TYPST_AMD64_SHA256="a6d077d0a95eed5a2eba715b2dae06be954f624ccbf85758a03f389ded33118c"
ARG TYPST_ARM64_SHA256="5aa8d74a3d906e60ea12a66ac2f37f8eef1b14cbad7182a745e393a10c23dcee"

ENV DEBIAN_FRONTEND=noninteractive \
    UV_INSTALL_DIR="/usr/local/bin" \
    UV_LINK_MODE=copy

WORKDIR /build

# Install build prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    xz-utils \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, reliable, reproducible dependency resolution
COPY --from=ghcr.io/astral-sh/uv:0.6.1 /uv /usr/local/bin/uv

# Download and verify Typst binary with SHA256 integrity check
RUN set -eux; \
    if [ "${TARGETARCH}" = "arm64" ]; then \
        TYPST_ARCH="aarch64"; \
        TYPST_SHA256="${TYPST_ARM64_SHA256}"; \
    else \
        TYPST_ARCH="x86_64"; \
        TYPST_SHA256="${TYPST_AMD64_SHA256}"; \
    fi; \
    ARCHIVE="typst-${TYPST_ARCH}-unknown-linux-musl.tar.xz"; \
    curl --proto '=https' --tlsv1.2 -fsSL \
        "https://github.com/typst/typst/releases/download/v${TYPST_VERSION}/${ARCHIVE}" \
        -o "/tmp/${ARCHIVE}"; \
    echo "${TYPST_SHA256}  /tmp/${ARCHIVE}" | sha256sum -c -; \
    tar -xJf "/tmp/${ARCHIVE}" -C /tmp; \
    install -m 755 "/tmp/typst-${TYPST_ARCH}-unknown-linux-musl/typst" /usr/local/bin/typst; \
    /usr/local/bin/typst --version; \
    rm -rf "/tmp/${ARCHIVE}" "/tmp/typst-${TYPST_ARCH}-unknown-linux-musl"

# Create target virtualenv
ENV VIRTUAL_ENV=/opt/venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy package specifications
COPY pyproject.toml uv.lock README.md /build/
COPY ubt /build/ubt

# Build and install package into virtualenv
RUN uv pip install --no-cache-dir /build

# ==============================================================================
# Stage 2: Runtime
# ==============================================================================
FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="Universal Book Translator" \
      org.opencontainers.image.description="All-in-One container for self-hosted book translation with Typst reflow and CJK fonts" \
      org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    # Standard UBT container storage layout
    UBT_DATA_DIR="/data" \
    UBT_DB_DIR="/data/dbs" \
    UBT_OUTPUT_DIR="/data/output" \
    UBT_ALLOWED_DIRS="/data" \
    # Safe container network defaults
    UBT_HOST="0.0.0.0" \
    UBT_PORT="8000"

WORKDIR /app

# Install runtime dependencies:
# 1. fonts-noto-cjk: Essential for Chinese/Japanese/Korean typography without tofu squares
# 2. fontconfig: System font indexing and discovery (fc-list, fc-cache)
# 3. libgl1, libglib2.0-0: Required by OpenCV, PyMuPDF, and image processing libraries
# 4. curl, ca-certificates: Required for healthchecks and secure provider API egress
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-noto-cjk \
    fontconfig \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
    && fc-cache -f -v \
    && rm -rf /var/lib/apt/lists/*

# Copy Typst compiler binary from builder stage
COPY --from=builder /usr/local/bin/typst /usr/local/bin/typst

# Copy virtualenv from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root user (UID 10001) for security hardening (least privilege)
RUN useradd --create-home --uid 10001 --shell /bin/bash ubt \
    && mkdir -p /data/dbs /data/output /data/input /data/models \
    && chown -R ubt:ubt /data /app /home/ubt

# Storage volume for state persistence (databases, input books, rendered outputs)
VOLUME ["/data"]

# API and Web Console port
EXPOSE 8000

# Healthcheck checking /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Copy container entrypoint
COPY --chown=ubt:ubt deploy/docker/entrypoint.sh /usr/local/bin/entrypoint.sh
USER root
RUN chmod +x /usr/local/bin/entrypoint.sh
USER ubt

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["api"]
