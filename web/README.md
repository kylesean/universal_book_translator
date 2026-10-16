# UBT Operator Console & Review Workbench

Frontend for **Universal Book Translator (UBT)**: a local-first Single Page Application (SPA) designed as an operator cockpit and post-editing workbench for high-fidelity bilingual book translation.

See also:
- [`DESIGN.md`](DESIGN.md) for the visual design manifesto, paper-light palette, and anti-AI-slop guidelines.
- [`../docs/UBT_OPERATOR_CONSOLE_PRD.md`](../docs/UBT_OPERATOR_CONSOLE_PRD.md) for the comprehensive product requirements.

---

## Architecture & Views

The console is built with **React 19**, **TypeScript**, **Vite**, **Tailwind CSS**, and **TanStack Query / Virtual**:

- **Wizard (`src/views/wizard`)**: Document drag-and-drop ingest, instant preflight inspection, budget estimation, and profile/preset configuration.
- **Mission Control (`src/views/mission-control`)**: Real-time compilation monitor with live SSE progress, stage indicators, log streaming, and artifact download / re-export.
- **Review Workbench (`src/views/review`)**: Professional CAT-style review interface with bilingual comparison, MQM / MTQE error flagging, live editing with ledger writeback under writer lock, and TM recycling.
- **Quality Gate (`src/views/quality-gate`)**: Verification report viewer displaying FastPass metrics, formula validation, and delivery verdicts.
- **Language Assets (`src/views/language-assets`)**: Book terminology (Bible) and Translation Memory (TM) browser and editor.
- **Doctor (`src/views/doctor`)**: Environment, toolchain (Typst, pdfium, etc.), and provider connection health checks.

---

## Development

### Prerequisites

- [Node.js](https://nodejs.org/) (>= 20)
- [pnpm](https://pnpm.io/) (>= 9)

### Running Dev Server

1. Start the backend API server:
   ```bash
   uv run ubt serve --port 8000
   ```

2. Start the Vite dev server:
   ```bash
   cd web
   pnpm install
   pnpm dev
   ```

Vite proxies `/api` and `/events` requests to `http://127.0.0.1:8000` (configured in `vite.config.ts`).

---

## Building & Packaging

The frontend is distributed as static assets embedded directly in the `ubt` Python package:

```bash
# Build SPA and sync dist/ into ubt/api/static
python scripts/build_console_assets.py
```

FastAPI automatically serves these assets at `/` when `ubt serve` or `ubt web` is invoked.

### Linting & Type Checking

```bash
pnpm lint       # Fast oxlint check
pnpm build      # tsc typecheck + vite production bundle
```
