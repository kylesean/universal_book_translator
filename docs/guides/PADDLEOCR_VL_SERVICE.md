# PaddleOCR-VL as a region reader (table / formula crops)

> **Scope** (2026-10-28): PaddleOCR-VL reads **crops**, not pages. It solves
> table data-cell recovery and formula transcription — the two things Docling
> does worst at the region level. Page-level OCR proofread (scanned or
> weak-text pages) is Unlimited-OCR-2's job; see [`OCR_DRIVERS.md`](OCR_DRIVERS.md)
> for that tier.

Docling's formula and table recognition is the ceiling on delivery quality: every
formula on a scientific paper degrades to a screenshot (measured: `formula_blocks:
0`), and its table reader fuses stacked rows into one cell. PaddleOCR-VL-1.6 is
the current leader on OmniDocBench (96.34 overall, formula CDM 97.53, table TEDS
94.76, Apache-2.0) — an order of magnitude better on exactly the two things that
fail.

## Measured on this project's documents (2026-10-17)

OmniDocBench averages hid a split this project has to care about: the two failure
modes it is chosen for are **not** the two it is best at. Numbers below are from
arXiv 2609.20519 (tables) and 2606.23344 (formulas), PaddleOCR-VL-1.6 through
`transformers` 5.20.1, RTX 4070 Ti, crops at 200–300 dpi.

| Task | Result | Verdict |
| --- | --- | --- |
| Table, 2-row merged header, 9 data rows (Table 1) | **90/90 data cells exact**; header spans correct | excellent |
| Table, grouped sub-headers, 6 rows (Table 4) | **54/54 cells exact** | excellent |
| Determinism | two runs, identical output | deterministic at `temperature=0` |
| Display formula `S_{i,j} = …` | `S_{i,j}`, `q_i^⊤`, `W_q^⊤`, all subscripts correct; **`d_h` → `d_b`** | good, one glyph error |
| Inline formula `Q = {q_i}_{i=1}^N ∈ ℝ^{N×d}` | exact under both `OCR:` and `Formula Recognition:` | excellent |
| Table **text** in header | `Recorded Token Traffic` → **`Attack`**, `Token Eff.` → **`EI.`** | 2 of 11 header labels wrong |
| Table **footnote** | `Note. API prices…` merged into the grid as a data row | needs post-filtering |
| Page-level `Formula Recognition:` | repeats one equation until the token cap (8192 tokens, 759 s) | **do not use per page** |
| Page-level `OCR:`, 1 page, no layout model | 509 vs 518 source words; every number and cross-reference correct | strong |
| Page with several tables | `Table 3`'s 9 rows collapsed into **one comma-separated line** | needs the layout model |

The last row is the important one: **the model is a per-region reader, not a page
parser.** It recognises a table or a formula beautifully when the crop is clean and
it degrades badly when the crop contains several structures. The official pipeline
runs PP-DocLayoutV2 first and sends each detected region through a task-specific
prompt — that layout stage is the missing piece in this project's integration.

Speed, measured end to end on this card (consumer GPU, batch 1): **~2.6 s per page**
for a clean page at 200 dpi once weights are loaded, against ~1.9 s/page for
Docling on the same document. A 73-page paper is roughly 3 minutes of vision time
on top of translation.

## Why this is a service, not a Python dependency

> **Correction (2026-10-17):** the section below was wrong on one point. The
> architecture *is* usable from `transformers` — but only from **v5+**
> (`transformers>=5.0.0`, verified with 5.20.1). On 4.57.6 it fails with
> `Unrecognized configuration class ...PaddleOCRVLConfig`, which is what led to
> the service-only conclusion. The service route is still the right production
> shape (it owns the layout model and batching), but a local `transformers`
> path now exists and is what produced the measurements above.

The supported paths are the `paddleocr` native API (a large new native dependency
alongside torch), an HTTP service, or `transformers>=5` for element-level
recognition. This project takes the service route for production: **no new
dependency in the main environment, no second inference stack, and the model can
run on a different machine from the one translating.**

## What already works

Nothing had to be written. `CloudOcrDriver` speaks the OpenAI vision protocol
(`/chat/completions` with `image_url`), which is exactly what vLLM serves. So:

```bash
# On the GPU box (12 GB+ free VRAM recommended):
vllm serve PaddlePaddle/PaddleOCR-VL-1.6 \
    --port 8185 \
    --trust-remote-code \
    --limit-mm-per-prompt image=1 \
    --max-model-len 16384

# On this machine, point the existing vision-OCR channel at it:
export UBT_OCR_ENDPOINT=http://127.0.0.1:8185/v1/chat/completions
export UBT_OCR_PROVIDER=vlm                  # explicit: never infer from the URL
export UBT_OCR_MODEL=PaddlePaddle/PaddleOCR-VL-1.6
export UBT_OCR_API_KEY=local-only            # vLLM ignores it; the client sends it
```

Then run with the VLM channel selected:

```bash
UBT_OCR_MODE=vlm ubt translate paper.pdf --pdf-engine docling
```

## Hardware, measured (not estimated)

| GPU | backend | pages/s | VRAM |
| --- | --- | --- | --- |
| RTX 3060 | vLLM | 0.53 | 11.8 GB |
| RTX 4090D | vLLM | 1.47 | 16.3 GB |
| A10 | FastDeploy | 1.15 | 21.8 GB |
| A100 | FastDeploy | 2.02 | 62.1 GB |

Model weights are ~2 GB in FP16 / ~1 GB INT8; the end-to-end pipeline (layout +
VLM + post-processing) is what costs 12–25 GB. **8 GB is the documented floor.**

On this development machine the practical ceiling is lower than the card: an
RTX 4070 Ti has 12.4 GB, but a local `llama-server` resident holds 4.2 GB and
the browser ~0.5 GB, leaving 5.9 GB free — below the 8 GB floor. Free the
resident first, or run the service elsewhere.

## Verifying the service before a full run

`scripts/check_ocr_endpoint.py` posts one rendered page and reports what came
back, so a misconfigured endpoint fails in seconds instead of after a 40-page
translation:

```bash
uv run python scripts/check_ocr_endpoint.py --pdf ~/Downloads/2606.23344v1.pdf --page 4
```

It checks four things and exits non-zero on any failure:

1. the endpoint answers at all;
2. it returns text (not an empty completion);
3. the returned text is *not* the English source echoing back (a model that
   refuses vision typically echoes the prompt);
4. the request completed inside a sane time budget.

## Trust boundary

`CloudOcrDriver` reports `measured_boxes=False` for the vision route: a language
model's coordinates are hallucinations, and the adapter refuses them in
recognition mode rather than anchoring text to invented boxes. That is why this
channel upgrades **formula and table markup**, not the geometric page layout —
layout still comes from the existing extractor.

## Why not make it the default

The default path must work on a CPU-only machine with no network. A VLM service
needs a GPU and a separate process, so it stays opt-in:

`auto` never selects it. Choose it explicitly with `UBT_OCR_MODE=vlm`, or let a
preset do so once you have the service running.

## Recommendation: adopt it as a *region* reader, not a page parser

The measurements say the model is worth adopting for exactly the two things
Docling does worst, **provided** it is fed one region at a time:

1. **Tables.** 144/144 data cells exact across two tables with merged headers,
   against Docling's known failure of fusing stacked rows. This alone removes most
   of the `_split_stacked_cell` / `_repair_*` workarounds.
2. **Formulas.** Subscripts and `⊤` survive — the exact defects Docling imposes
   (`{q_i}_{i=1}^N` scrambling, transpose loss). One glyph error in eight
   (`d_h` → `d_b`), so formula output still needs the existing structural check
   rather than being trusted outright.

What it must **not** replace:

* **Page layout.** Docling still supplies reading order, geometry and page
  segmentation. Feeding a whole page to PaddleOCR-VL collapses multi-table pages.
* **Table header text.** Two of eleven header labels were misread; headers are
  also what the leak metric and the translator key on. Prefer the source text-layer
  header where it exists, or cross-check against Docling's header row.
* **Table footnotes.** They arrive inside the grid and must be split back out.

The integration therefore looks like: Docling detects the region and its bbox →
PaddleOCR-VL reads that crop for `Table Recognition:` / `Formula Recognition:` →
the existing QE and renderer consume the result unchanged. That is a bounded
change behind the adapter seam, not a second pipeline.

## The layout model is the missing piece

Per-page use of the model alone (no PP-DocLayoutV2) was measured and is not good
enough to replace Docling: on a page with three tables, Table 3's nine rows came
back as one comma-separated line, and `Formula Recognition:` on a whole page loops
until the token cap. Both are layout failures, not recognition failures. Any
adoption plan needs the layout stage — either PP-DocLayoutV2 beside the model, or
Docling's existing detection used as the region source (the cheaper option, and the
one that reuses what already works).
