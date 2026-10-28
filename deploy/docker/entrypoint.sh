#!/usr/bin/env bash
set -eo pipefail

# Universal Book Translator (UBT) Container Entrypoint
# Dispatches between REST API / Web Console, CLI commands, Queue Worker, and MCP server.

# Default storage directories under /data
mkdir -p "${UBT_DB_DIR:-/data/dbs}" "${UBT_OUTPUT_DIR:-/data/output}" "${UBT_ALLOWED_DIRS:-/data}"

# In containerized environments, provide safe local development defaults
if [ -z "${UBT_API_KEY}" ] && [ "${UBT_ALLOW_NO_AUTH:-1}" = "1" ]; then
    export UBT_ALLOW_NO_AUTH="1"
    export UBT_ALLOW_INSECURE_BIND="1"
fi

COMMAND="${1:-api}"

case "${COMMAND}" in
    api|serve)
        shift 2>/dev/null || true
        HOST="${UBT_HOST:-0.0.0.0}"
        PORT="${UBT_PORT:-8000}"
        echo "Starting UBT API & Web Console on http://${HOST}:${PORT}..."
        exec ubt-api --host "${HOST}" --port "${PORT}" "$@"
        ;;
    doctor)
        shift
        echo "Running UBT system and environment diagnostics..."
        exec ubt doctor "$@"
        ;;
    translate)
        shift
        exec ubt translate "$@"
        ;;
    worker)
        shift
        echo "Starting UBT background queue worker..."
        exec ubt worker "$@"
        ;;
    mcp)
        shift
        exec ubt-mcp "$@"
        ;;
    *)
        # Passthrough arbitrary commands (e.g. bash, python, ubt assess)
        exec "$@"
        ;;
esac
