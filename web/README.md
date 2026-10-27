# UBT Operator Console & Review Workbench

Frontend for **Universal Book Translator (UBT)**: a local-first Single Page Application (SPA) designed as an operator cockpit and post-editing workbench for high-fidelity bilingual book translation.

See also:
- [`DESIGN.md`](DESIGN.md) for the visual design manifesto, light/dark palette, and anti-AI-slop guidelines.
- [`../ARCHITECTURE.md`](../ARCHITECTURE.md) for the engine architecture the console drives.

---

## Architecture & Views

The console is built with **React 19**, **TypeScript**, **Vite**, **Tailwind CSS**,
and **shadcn/ui** primitives (Radix UI + CVA). The application shell is an inset
sidebar with a mobile Sheet drawer and a command palette; Geist and Geist Mono
are self-hosted. Route-level code splitting keeps each workbench independent.

- **Wizard (`src/views/wizard`)**: Document drag-and-drop ingest, instant preflight inspection, budget estimation, and profile/preset configuration.
- **Mission Control (`src/views/mission-control`)**: Real-time compilation monitor with live SSE progress, stage indicators, log streaming, and artifact download / re-export.
- **Review Workbench (`src/views/review`)**: Professional CAT-style review interface with bilingual comparison, MQM / MTQE error flagging, continuous bilingual proofing (`view=flow`), live editing with ledger writeback under writer lock, and TM recycling.
- **Quality Gate (`src/views/quality-gate`)**: Verification report viewer displaying FastPass metrics, formula validation, and delivery verdicts.
- **Language Assets (`src/views/language-assets`)**: Shared glossary and translation-memory browser, editor, and importer.
- **Doctor (`src/views/doctor`)**: Environment, toolchain (Typst, pdfium, etc.), and provider connection health checks.

`/wizard` is the only translation intake. UBT is a translation compiler and
proofing console, not a consumer e-reader: finished books are previewed or
downloaded from the Quality Gate's deliverables.

Reusable application components live in `src/components/ui`. Components installed
through the shadcn CLI keep their registry names and structure; page-specific
composition stays in the view that owns it.

---

## Development

### Prerequisites

- [Node.js](https://nodejs.org/) (>= 20)
- [pnpm](https://pnpm.io/) (>= 9)

### Running Dev Server

1. Start the backend API server:
   ```bash
   uv run ubt console --port 8000
   ```

2. Start the Vite dev server:
   ```bash
   cd web
   pnpm install
   pnpm dev
   ```

Vite proxies data requests to `http://127.0.0.1:8000` (configured in
`vite.config.ts`). `/jobs`, `/assets`, and `/system` are both SPA routes and API
prefixes; the dev proxy bypasses backend proxying for browser HTML navigations so
direct route loads serve the current SPA instead of a stale embedded bundle.

---

## Building & Packaging

The frontend is distributed as static assets embedded directly in the `ubt` Python package:

```bash
# Build SPA and sync dist/ into ubt/api/static
python scripts/build_console_assets.py
```

FastAPI automatically serves these assets at `/` when `ubt console` or `ubt web` is invoked.

### Linting & Type Checking

```bash
pnpm lint       # Fast oxlint check
pnpm build      # tsc typecheck + vite production bundle
```

The production build must also be synced before it is served by FastAPI:

```bash
python scripts/build_console_assets.py
```
