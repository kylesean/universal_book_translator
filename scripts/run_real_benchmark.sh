#!/usr/bin/env bash
# =============================================================================
# UBT Real-Model & End-to-End Evaluation Runner
# Drives real LLM API, the local llama-swap MT backend, and CometKiwi QE benchmarks
# Reference: docs/evaluation-and-comparison-guide.md
# =============================================================================

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "================================================================================"
echo " Universal Book Translator (UBT) - Real-World Benchmark Driver"
echo "================================================================================"

# 1. Probe Local Capabilities
echo "[1/4] Probing local environment dependencies & weights..."

HAS_POPPLER_SVG=false
if command -v pdftocairo >/dev/null 2>&1 && command -v pdftotext >/dev/null 2>&1; then
    HAS_POPPLER_SVG=true
fi

HAS_TYPST=false
if command -v typst >/dev/null 2>&1; then
    HAS_TYPST=true
fi

# The local MT backend lives behind the llama-swap gateway (:9090), which loads
# the translategemma backend on demand. The Ollama daemon on :11434 was retired
# 2026-09-23 (unit disabled, weights deleted): probing it would skip this tier
# forever. Override the endpoint with UBT_LIVE_MT_BASE_URL.
LOCAL_LLM_BASE_URL="${UBT_LIVE_MT_BASE_URL:-http://127.0.0.1:9090}"
HAS_LOCAL_LLM=false
HAS_MT_MODEL=false
if curl -s "${LOCAL_LLM_BASE_URL}/v1/models" >/dev/null 2>&1; then
    HAS_LOCAL_LLM=true
    if curl -s "${LOCAL_LLM_BASE_URL}/v1/models" | grep -q "translategemma"; then
        HAS_MT_MODEL=true
    fi
fi

HAS_COMET_WEIGHTS=false
HF_HUB="${HOME}/.cache/huggingface/hub"
# One directory level: models--<org>--<repo>/snapshots/<rev>/checkpoints/model.ckpt.
# An extra `*/` here made this probe report "⚠️ Missing" while 2.2 GB of weights
# (and the sibling hparams.yaml find_cometkiwi_checkpoint requires) sat in the
# cache — which silently skipped --qe-calib.
if compgen -G "${HF_HUB}/models--*cometkiwi*/snapshots/*/checkpoints/model.ckpt" > /dev/null; then
    HAS_COMET_WEIGHTS=true
fi

# The names UBTConfig declares as credentials (ubt/core/config.py api_key
# aliases). OPENCODE_API_KEY was missing, so a machine authenticated only through
# it reported "⚠️ Unset" and --compare-live self-skipped.
HAS_LLM_KEY=false
if [ -n "${UBT_LLM_API_KEY}" ] || [ -n "${OPENCODE_API_KEY}" ] \
    || [ -n "${OPENAI_API_KEY}" ] || [ -n "${DEEPSEEK_API_KEY}" ] \
    || [ -n "${ANTHROPIC_API_KEY}" ] || [ -n "${GEMINI_API_KEY}" ] \
    || [ -n "${UBT_OPENCODE_SESSION_ID}" ]; then
    HAS_LLM_KEY=true
fi

echo "  - Poppler (pdftocairo / pdftotext) : $( [ "${HAS_POPPLER_SVG}" = true ] && echo "✅ Ready (SVG vector active)" || echo "⚠️ Missing (Falls back to raster)" )"
echo "  - Typst Vector Compiler            : $( [ "${HAS_TYPST}" = true ] && echo "✅ Ready" || echo "❌ Not Found" )"
echo "  - Local LLM Gateway (llama-swap)   : $( [ "${HAS_LOCAL_LLM}" = true ] && echo "✅ Online (${LOCAL_LLM_BASE_URL})" || echo "⚠️ Offline" )"
echo "  - TranslateGemma Model (llama-swap): $( [ "${HAS_MT_MODEL}" = true ] && echo "✅ Loaded (translategemma:4b)" || echo "⚠️ Not served (add it to llama-swap config.yaml)" )"
echo "  - CometKiwi Neural QE Checkpoint   : $( [ "${HAS_COMET_WEIGHTS}" = true ] && echo "✅ Found in HF cache" || echo "⚠️ Missing in ~/.cache/huggingface/hub" )"
echo "  - Cloud LLM Authentication         : $( [ "${HAS_LLM_KEY}" = true ] && echo "✅ Configured via environment" || echo "⚠️ Unset (Mock provider will be used)" )"

MODE="${1:---summary}"

# 2. Execution Routing
echo ""
echo "[2/4] Executing targeted benchmarks for mode: ${MODE}"

case "${MODE}" in
    --live-mt)
        if [ "${HAS_LOCAL_LLM}" = true ] && [ "${HAS_MT_MODEL}" = true ]; then
            echo "Running Live Local MT Tier Acceptance Test..."
            # -o addopts="": these files carry pytestmark=slow, which the default
            # marker filter deselects — pytest would exit 5 ("nothing collected")
            # and `set -e` would abort the mode before it ran.
            uv run pytest tests/integration/test_mt_tier_live.py -v -s -o addopts=""
        else
            echo "⚠️ Skipping --live-mt: llama-swap is not serving translategemma:4b."
        fi
        ;;
    --qe-calib)
        if [ "${HAS_COMET_WEIGHTS}" = true ]; then
            echo "Running L1/L2 QE Consistency & Calibration Test..."
            uv run pytest tests/integration/test_qe_calibration.py -v -s -o addopts=""
        else
            echo "⚠️ Skipping --qe-calib: CometKiwi checkpoint missing."
        fi
        ;;
    --compare-live)
        if [ "${HAS_LLM_KEY}" = true ] && [ "${HAS_MT_MODEL}" = true ]; then
            echo "Running MT vs Cloud LLM Paired Comparison..."
            uv run pytest tests/integration/test_mt_vs_llm_compare.py -v -s -o addopts=""
        else
            echo "⚠️ Skipping --compare-live: Requires both LLM Key/Session and a local MT model."
        fi
        ;;
    --all)
        echo "Running complete live suite where prerequisites are met..."
        uv run pytest tests/integration/ -v -s -o addopts=""
        ;;
    --summary|*)
        echo "Diagnostic mode complete. To execute specific live benchmarks, run with:"
        echo "  ./scripts/run_real_benchmark.sh --live-mt      # Real TranslateGemma local MT run"
        echo "  ./scripts/run_real_benchmark.sh --qe-calib     # Neural CometKiwi score calibration"
        echo "  ./scripts/run_real_benchmark.sh --compare-live # Paired comparison against Cloud LLM"
        echo "  ./scripts/run_real_benchmark.sh --all          # Run all live integration tests"
        ;;
esac

echo ""
echo "[3/4] Running zero-token deterministic regression guard..."
# test_readme_case_count.py was removed in 9397326; the KPI-schema and shared
# QE-score-policy tests are the deterministic guards that survived it.
uv run pytest tests/unit/test_metrics.py tests/unit/test_qe_score_policy.py -q

echo ""
echo "[4/4] Benchmark workflow completed."
echo "================================================================================"
