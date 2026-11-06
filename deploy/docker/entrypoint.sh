#!/usr/bin/env bash
set -eo pipefail

# Universal Book Translator (UBT) Container Entrypoint
# Dispatches between REST API / Web Console, CLI commands, Queue Worker, and MCP server.

# Default storage directories under /data
mkdir -p "${UBT_DB_DIR:-/data/dbs}" "${UBT_OUTPUT_DIR:-/data/output}" "${UBT_ALLOWED_DIRS:-/data}"

# Container defaults stay fail-closed: without UBT_API_KEY the API refuses to
# start (ubt/api/security.py), so an accidental `docker run` cannot publish an
# unauthenticated console. The old branch here *widened* both escapes whenever
# no key was set, which turned the default deployment into an open LAN service.
# Only the `api`/`serve` command needs the key; batch commands (translate,
# doctor, mcp stdio) are local and must keep working without one.
COMMAND="${1:-api}"

if [ "${COMMAND}" = "api" ] || [ "${COMMAND}" = "serve" ]; then
    if [ -z "${UBT_API_KEY}" ]; then
        if [ "${UBT_ALLOW_NO_AUTH:-0}" != "1" ]; then
            echo "error: UBT_API_KEY is not set." >&2
            echo "       Set UBT_API_KEY=<token> to start the console," >&2
            echo "       or UBT_ALLOW_NO_AUTH=1 for a keyless local-only instance" >&2
            echo "       (pair it with a 127.0.0.1 host port mapping)." >&2
            exit 1
        fi
        # Keyless was an explicit operator decision. The container still has to
        # listen on 0.0.0.0 for the port mapping to work, and the app refuses a
        # non-loopback bind without full isolation, so follow the same explicit
        # decision here instead of making the operator discover a second flag.
        if [ "${UBT_ALLOW_INSECURE_BIND:-0}" != "1" ]; then
            echo "warning: running keyless on 0.0.0.0 (UBT_ALLOW_NO_AUTH=1)." >&2
            echo "         Publish it on 127.0.0.1 only; never on a shared network." >&2
            export UBT_ALLOW_INSECURE_BIND="1"
        fi
    fi
fi

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
