#!/usr/bin/env python3
"""Air-gapped CometKiwi / MTQE subprocess scoring script.

One-shot mode (``--score-stdin``): reads a JSON array of ``{"src": ..., "mt":
...}`` objects from standard input and outputs
``{"scores": [...], "engine": "neural"|"heuristic_fallback"}``. The engine
label is load-bearing: `SubprocessQERunner` disables best-of-n reranking
whenever the batch was not actually scored by the neural model, so a fallback
batch can never masquerade as a calibrated measurement.

Serving mode (``--serve``): reads JSON-lines requests
``{"id": N, "pairs": [...]}`` from stdin and writes one
``{"id": N, "scores": [...], "engine": ...}`` response line per request. The
model weights load once and stay resident across batches; stdin EOF ends the
session (the parent dying therefore never orphans this process).

Designed to be executed in an isolated subprocess by `SubprocessQERunner` via JSON IPC,
protecting the main translation pipeline from PyTorch / CUDA memory leaks or C++ crashes.
"""

import argparse
import contextlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _fallback_score(pairs: list[dict[str, Any]]) -> list[float]:
    return [calculate_fallback_score(str(p.get("src", "")), str(p.get("mt", ""))) for p in pairs]


def build_scorer(
    args: argparse.Namespace,
) -> tuple[Callable[[list[dict[str, Any]]], list[float]], str]:
    """Return (score_fn, engine) for the configured model.

    ``engine`` is exactly what the parent trusts: "neural" only when the
    CometKiwi weights actually imported and loaded; "heuristic_fallback" for
    the mock switch or an unimportable comet/torch (stderr carries the hint).
    """
    use_mock = args.mock or os.getenv("UBT_COMET_MOCK", "").lower() in ("1", "true", "yes")
    if use_mock:
        return _fallback_score, "heuristic_fallback"
    try:
        import torch
        from comet import load_from_checkpoint

        gpus = 1 if torch.cuda.is_available() else 0
        model_path = resolve_checkpoint(args.model)
        sys.stderr.write(f"COMET loading checkpoint: {model_path} (GPU: {bool(gpus)})\n")
        model = load_from_checkpoint(model_path)
    except ImportError:
        sys.stderr.write(
            "Warning: 'unbabel-comet' or 'torch' package not installed in python environment.\n"
            "To use neural CometKiwi QE, install them in the worker environment: pip install unbabel-comet torch\n"
            "Falling back to deterministic heuristic scorer.\n"
        )
        return _fallback_score, "heuristic_fallback"
    except Exception as exc:
        # A corrupt/undownloadable checkpoint, a CUDA OOM, or an incompatible
        # torch build must degrade to the deterministic scorer this module
        # documents — not propagate to ``serve``'s exit(3), which kills the
        # whole job's QE/repair pass.
        sys.stderr.write(
            f"Warning: COMET checkpoint load failed ({type(exc).__name__}: {exc}); "
            "falling back to deterministic heuristic scorer.\n"
        )
        return _fallback_score, "heuristic_fallback"

    def score(pairs: list[dict[str, Any]]) -> list[float]:
        data = [{"src": str(p.get("src", "")), "mt": str(p.get("mt", ""))} for p in pairs]
        model_output = model.predict(
            data,
            batch_size=args.batch_size,
            gpus=gpus,
            progress_bar=False,
            num_workers=0,
        )
        return [round(float(s), 4) for s in model_output.scores]

    return score, "neural"


def serve(args: argparse.Namespace) -> None:
    """JSON-lines request loop; one response per request, exit on stdin EOF."""
    try:
        score, engine = build_scorer(args)
    except Exception as exc:
        sys.stderr.write(f"COMET server setup failed: {exc}\n")
        sys.exit(3)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id: Any = None
        try:
            request = json.loads(line)
            req_id = request.get("id")
            cmd = request.get("cmd", "")
            if cmd == "quit":
                return
            pairs = request.get("pairs", [])
            if not isinstance(pairs, list):
                raise ValueError("'pairs' must be a JSON list")
            scores = score(pairs)
            reply: dict[str, Any] = {"id": req_id, "scores": scores, "engine": engine}
        except Exception as exc:
            # One bad request must not kill the session: answer with an error
            # envelope so the parent can distinguish protocol failure from a
            # model failure and keep queued requests flowing.
            reply = {"id": req_id, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def calculate_fallback_score(src: str, mt: str) -> float:
    """Deterministic heuristic score when PyTorch / COMET is absent or mock requested."""
    if not mt or not mt.strip():
        return 0.0
    # Match the FastPass leak definition (fast_pass.py): the *opening* tag is
    # what a leaked scaffold carries; checking only the closing tag let a
    # truncated '<translation>…' leak score ~0.85 as if it were clean.
    if "<issues>" in mt or "<translation>" in mt or "</translation>" in mt:
        return 0.20
    ratio = min(len(mt), len(src)) / max(1, max(len(mt), len(src)))
    score = min(1.0, max(0.1, 0.85 * (0.8 + 0.2 * ratio)))
    return round(float(score), 4)


def resolve_checkpoint(model_arg: str) -> str:
    """Resolve local cached checkpoint or download if needed.

    Prefers locally cached snapshots (e.g. from ben-xl8 or Unbabel mirrors)
    without requiring online HuggingFace Hub authentication.
    """
    from comet import download_model

    direct_path = Path(model_arg).expanduser()
    if direct_path.is_file():
        # Comet's load_from_checkpoint expects parent.parent to hold hparams.yaml
        if (direct_path.parent / "hparams.yaml").is_file():
            sub_dir = direct_path.parent / "checkpoints"
            sub_dir.mkdir(exist_ok=True)
            sub_file = sub_dir / direct_path.name
            if not sub_file.exists():
                with contextlib.suppress(OSError):
                    sub_file.symlink_to(direct_path)
            if sub_file.is_file():
                return str(sub_file)
        return str(direct_path)

    if direct_path.is_dir():
        ckpt_candidate = direct_path / "checkpoints" / "model.ckpt"
        if ckpt_candidate.is_file():
            return str(ckpt_candidate)
        ckpt_candidate = direct_path / "model.ckpt"
        if ckpt_candidate.is_file():
            return resolve_checkpoint(str(ckpt_candidate))

    env_path_str = os.getenv("UBT_COMET_MODEL_PATH")
    if env_path_str:
        env_path = Path(env_path_str).expanduser()
        # Guard against self-re-entry: when the caller's model_arg already IS the
        # env dir (and it holds no checkpoint at the two known locations), the
        # recursive call re-enters this same branch with the same path and never
        # reaches the download fallback — a RecursionError surfaced as a bare
        # exit code 3.
        if env_path.exists() and env_path != direct_path:
            return resolve_checkpoint(str(env_path))

    # Check local Hugging Face cache for known mirrors/repos
    candidates = [model_arg]
    if "wmt22-cometkiwi-da" in model_arg:
        candidates = ["ben-xl8/wmt22-cometkiwi-da", "Unbabel/wmt22-cometkiwi-da", model_arg]

    for cand in candidates:
        try:
            resolved = download_model(cand, local_files_only=True)
            if resolved and Path(str(resolved)).is_file():
                return str(resolved)
        except Exception:
            # Local-resolution probe: a missing candidate is expected, and the
            # final download below raises if nothing resolves.
            pass

    return str(download_model(model_arg))


def main() -> None:
    parser = argparse.ArgumentParser(description="CometKiwi air-gapped MTQE subprocess scoring")
    parser.add_argument(
        "--score-stdin",
        action="store_true",
        help="Read input JSON pairs from stdin and output scores JSON to stdout",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Unbabel/wmt22-cometkiwi-da",
        help="HuggingFace / COMET model checkpoint name or local path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for model evaluation",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force fallback mock scoring without loading PyTorch COMET weights",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve JSON-lines {id, pairs} requests from stdin with one response line each",
    )

    args = parser.parse_args()

    if args.serve:
        serve(args)
        return

    if not args.score_stdin:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # Read all input from stdin
    try:
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            sys.stdout.write(json.dumps({"scores": [], "engine": "unknown"}))
            sys.stdout.flush()
            return
        pairs: list[dict[str, Any]] = json.loads(raw_input)
    except Exception as exc:
        sys.stderr.write(f"Failed to parse stdin JSON payload: {exc}\n")
        sys.exit(2)

    if not isinstance(pairs, list):
        sys.stderr.write("Input payload must be a JSON list of objects\n")
        sys.exit(2)

    try:
        score, engine = build_scorer(args)
    except Exception as exc:
        sys.stderr.write(f"COMET neural evaluation error: {exc}\n")
        sys.exit(3)

    sys.stdout.write(json.dumps({"scores": score(pairs), "engine": engine}))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
