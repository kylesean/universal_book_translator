#!/usr/bin/env bash
# =============================================================================
# Formula engine matrix runner
#
# Runs tests/fixtures/synthetic-duo.pdf through every math_backend / formula_render
# combination, then prints a Markdown summary of page counts, witness
# fallbacks and wall time per scenario.
#
# Default is --dry-run (MockProvider, zero API cost). Use --real to translate
# with the configured provider; the ledger is reused per scenario job id, so a
# --real run may resume instead of re-translating.
#
# Companion assets:
#   tests/integration/test_formula_engine_matrix.py  (nightly regression)
#
# This is the cheapest way to fill the repo's wall-clock gap: it is mock by
# default (zero API cost) and already times every scenario. Commit
# ${OUT_DIR}/summary.md alongside the run that produced it.
#
# Usage:
#   scripts/formula_matrix.sh                 # mock, full chapter
#   scripts/formula_matrix.sh --pages 1-6     # mock, subset
#   scripts/formula_matrix.sh --real          # real provider (token cost)
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SOURCE="tests/fixtures/synthetic-duo.pdf"
OUT_DIR="output/formula-matrix"
PAGES=""
REAL=0
JOB_PREFIX="matrix"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --real) REAL=1; shift ;;
        --pages) PAGES="${2:-}"; shift 2 ;;
        --source) SOURCE="${2:-}"; shift 2 ;;
        --out-dir) OUT_DIR="${2:-}"; shift 2 ;;
        --job-prefix) JOB_PREFIX="${2:-}"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -f "${SOURCE}" ]] || { echo "source PDF not found: ${SOURCE}" >&2; exit 2; }
mkdir -p "${OUT_DIR}"

COMMON=(translate "${SOURCE}" --source-lang en --target-lang zh)
if [[ ${REAL} -eq 0 ]]; then
    COMMON+=(--dry-run)
    echo "mode: mock (--dry-run); use --real for provider translation"
else
    echo "mode: REAL provider translation — token cost applies"
fi
[[ -n "${PAGES}" ]] && COMMON+=(--pages "${PAGES}")

# name|backend|formula_render|hide_node
SCENARIOS=(
    "mathjax|mathjax|witness|0"
    "image|image|witness|0"
    "typst-witness|typst|witness|0"
    "typst-image|typst|image|0"
    "typst-native|typst|native|0"
    "mathjax-no-node|mathjax|witness|1"
)

SUMMARY="${OUT_DIR}/summary.md"
{
    echo "# Formula engine matrix"
    echo
    echo "- source: \`${SOURCE}\`"
    echo "- mode: $([[ ${REAL} -eq 0 ]] && echo "mock (--dry-run)" || echo "real provider")"
    echo "- pages: ${PAGES:-all}"
    echo "- generated: $(date -Iseconds)"
    echo
    echo "| scenario | backend | render | pages | witness fallbacks | wall |"
    echo "|---|---|---|---|---|---|"
} > "${SUMMARY}"

failures=0
for scenario in "${SCENARIOS[@]}"; do
    IFS='|' read -r name backend render hide_node <<< "${scenario}"
    job="${JOB_PREFIX}-${name}"
    output="${OUT_DIR}/${name}.pdf"
    report="${OUT_DIR}/${name}_quality_report.json"
    echo "=== ${name}: backend=${backend} render=${render} job=${job}"
    start=$(date +%s)
    if [[ "${hide_node}" == "1" ]]; then
        # Filtering every PATH entry when node is absent leaves an empty value,
        # and under `set -euo pipefail` the failing grep pipeline aborts the
        # whole matrix. Only filter when node exists; otherwise keep PATH.
        if command -v node >/dev/null 2>&1; then
            node_dir="$(dirname "$(command -v node)")"
            clean_path="$(echo "${PATH}" | tr ':' '\n' | grep -vF "${node_dir}" | paste -sd: - || true)"
        else
            node_dir="(none)"
            clean_path="${PATH}"
        fi
        if env PATH="${clean_path}" command -v node >/dev/null 2>&1; then
            echo "  warning: node still resolvable without ${node_dir}; degradation test is not isolated" >&2
        fi
        env PATH="${clean_path}" uv run ubt "${COMMON[@]}" \
            --math-backend "${backend}" --formula-render "${render}" \
            --job-id "${job}" --output "${output}" --fresh || failures=$((failures + 1))
    else
        uv run ubt "${COMMON[@]}" \
            --math-backend "${backend}" --formula-render "${render}" \
            --job-id "${job}" --output "${output}" --fresh || failures=$((failures + 1))
    fi
    elapsed=$(( $(date +%s) - start ))
    pages="-"
    [[ -f "${output}" ]] && pages="$(pdfinfo "${output}" 2>/dev/null | awk '/^Pages:/ {print $2}')"
    witness="$(python3 - "${report}" <<'PY'
import json, sys
try:
    report = json.load(open(sys.argv[1], encoding="utf-8"))
    print(len(report.get("formula_witness_fallbacks") or []))
except (OSError, ValueError):
    print("-")
PY
)"
    {
        echo "| ${name} | ${backend} | ${render} | ${pages} | ${witness} | ${elapsed}s |"
    } >> "${SUMMARY}"
done

echo
cat "${SUMMARY}"
echo
echo "full reports: ${OUT_DIR}/*_quality_report.json"
if [[ ${failures} -gt 0 ]]; then
    echo "${failures} scenario(s) failed" >&2
    exit 1
fi
