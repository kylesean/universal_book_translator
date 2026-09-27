"""Shared probes for live-service integration tests (local MT, CometKiwi QE).

Everything here degrades to skip-markers when the local service is absent,
so CI stays green without the local model stack or HF checkpoints. Nothing in
this module is collected as a test (underscore prefix).

The MT tier runs on the llama-swap gateway (``127.0.0.1:9090``), which loads
the ``translategemma:4b`` backend on demand. The Ollama daemon that used to
serve it was retired (unit disabled, weights deleted), so probing
``:11434`` would silently skip this tier forever. Point ``UBT_LIVE_MT_BASE_URL``
somewhere else when the stack runs on another host or port.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import zipfile
from pathlib import Path

import pytest

from ubt.core.qe.mt_gate import count_sentences

LOCAL_MT_BASE_URL = os.environ.get("UBT_LIVE_MT_BASE_URL", "http://127.0.0.1:9090").rstrip("/")
LOCAL_MT_API_BASE = f"{LOCAL_MT_BASE_URL}/v1"
MT_MODEL = "translategemma:4b"

BASELINES_DIR = Path(__file__).parents[1] / "baselines"
CORPUS_EPUBS = (
    BASELINES_DIR / "call-of-the-wild" / "call_of_the_wild.epub",
    BASELINES_DIR / "standard-alice" / "standard-alice.epub",
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_HAS_LETTER = re.compile(r"[A-Za-z]")


def local_mt_has_model(model: str = MT_MODEL) -> bool:
    """True when the local llama-swap gateway serves ``model``.

    Exact ``id`` match (not a substring like the retired Ollama ``name`` walk):
    a loose match would accept ``translategemma-4b`` aliases and any future
    ``translategemma:4b-q8`` sibling, then hand the tier a model it never
    declared.
    """
    try:
        with urllib.request.urlopen(f"{LOCAL_MT_API_BASE}/models", timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return any(model == (m.get("id") or "") for m in data.get("data", []))
    except Exception:
        return False


requires_local_mt = pytest.mark.skipif(
    not local_mt_has_model(),
    reason=f"local gateway not serving {MT_MODEL} at {LOCAL_MT_BASE_URL} "
    "(start llama-swap: systemctl --user start llama-swap)",
)


def find_cometkiwi_checkpoint() -> Path | None:
    """Locate a local CometKiwi checkpoint (Unbabel or ben-xl8 mirror)."""
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    for mirror in ("models--Unbabel--wmt22-cometkiwi-da", "models--ben-xl8--wmt22-cometkiwi-da"):
        for ckpt in sorted((hub / mirror).glob("snapshots/*/checkpoints/model.ckpt")):
            if (ckpt.parent.parent / "hparams.yaml").is_file():
                return ckpt
    return None


requires_cometkiwi = pytest.mark.skipif(
    find_cometkiwi_checkpoint() is None,
    reason="no local CometKiwi checkpoint in ~/.cache/huggingface/hub",
)


def live_llm_configured() -> bool:
    """True when the config ladder resolves a real outbound LLM credential.

    This marker gates the only test that spends real money, so it must be opted
    into: a bare third-party variable (``OPENAI_API_KEY`` and friends) is inert
    on its own, and only ``UBT_LLM_API_KEY`` or a selected provider's
    ``api_key_env`` supplies the credential. A malformed provider name must not
    break collection, so the probe degrades to "not configured".
    """
    from ubt.core.config import MOCK_API_KEY, UBTConfig

    try:
        key = UBTConfig.from_env().api_key.get_secret_value()
    except Exception:
        return False
    return bool(key) and key != MOCK_API_KEY


requires_live_llm = pytest.mark.skipif(
    not live_llm_configured(),
    reason="no explicit LLM credential (set UBT_LLM_API_KEY); implicit credential "
    "resolution is never billed",
)


#: Credential/base-url env captured at import time — before the autouse
#: ``hermetic_config`` fixture in tests/conftest.py strips every ``UBT_*`` var for
#: each test body. ``conftest`` re-applies it through the ``live_llm_env`` fixture
#: so a configured live run actually reaches the configured endpoint instead of
#: falling back to ``mock-key`` against ``api.openai.com``.
LIVE_ENV_KEYS = (
    "UBT_LLM_API_KEY",
    "OPENAI_API_KEY",
    "UBT_BASE_URL",
    "UBT_DRAFT_MODEL",
    "UBT_REPAIR_MODEL",
    "UBT_API_MODE",
    "UBT_PROVIDER",
)
LIVE_ENV_SNAPSHOT = {key: os.environ[key] for key in LIVE_ENV_KEYS if os.environ.get(key)}


def sample_corpus_sentences(
    limit_per_book: int = 12, min_chars: int = 20, max_chars: int = 240
) -> list[str]:
    """Deterministic single-sentence en samples from local baseline EPUBs."""
    from bs4 import BeautifulSoup

    sampled: list[str] = []
    seen: set[str] = set()
    for epub_path in CORPUS_EPUBS:
        if not epub_path.exists():
            continue
        got = 0
        with zipfile.ZipFile(epub_path) as zf:
            names = sorted(
                n for n in zf.namelist() if n.lower().endswith((".xhtml", ".html", ".htm"))
            )
            for name in names:
                soup = BeautifulSoup(zf.read(name).decode("utf-8", errors="ignore"), "html.parser")
                for p in soup.find_all("p"):
                    for sent in _SENT_SPLIT.split(p.get_text(" ", strip=True)):
                        s = sent.strip()
                        if (
                            min_chars <= len(s) <= max_chars
                            and count_sentences(s) == 1
                            and _HAS_LETTER.search(s)
                            and s not in seen
                        ):
                            seen.add(s)
                            sampled.append(s)
                            got += 1
                            if got >= limit_per_book:
                                break
                    if got >= limit_per_book:
                        break
                if got >= limit_per_book:
                    break
    return sampled
