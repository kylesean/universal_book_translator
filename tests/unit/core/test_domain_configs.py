"""Unit tests for typed sub-domain configuration models (RouterConfig, TypstConfig, QualityConfig, EngineConfig)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ubt.core.config import (
    EngineConfig,
    QualityConfig,
    RouterConfig,
    TypstConfig,
    UBTConfig,
)

pytestmark = pytest.mark.fast


def test_router_config_standalone_validation() -> None:
    router_cfg = RouterConfig(
        draft_model="gpt-4o",
        repair_model="gpt-4o-mini",
        rate_limit_rpm=120,
        rate_limit_tpm=200_000,
        api_timeout=60.0,
    )
    assert router_cfg.draft_model == "gpt-4o"
    assert router_cfg.repair_model == "gpt-4o-mini"
    assert router_cfg.rate_limit_rpm == 120
    assert router_cfg.rate_limit_tpm == 200_000
    assert router_cfg.api_timeout == 60.0
    assert router_cfg.prompt_caching_enabled is True

    # Validation: api_timeout must be > 0
    with pytest.raises(ValidationError):
        RouterConfig(api_timeout=-1.0)


def test_typst_config_standalone_validation() -> None:
    typst_cfg = TypstConfig(
        dual_mode="mono",
        font_family="Noto Serif CJK SC",
        spill_policy="appendix",
        layout_columns=2,
    )
    assert typst_cfg.dual_mode == "mono"
    assert typst_cfg.font_family == "Noto Serif CJK SC"
    assert typst_cfg.spill_policy == "appendix"
    assert typst_cfg.layout_columns == 2
    assert typst_cfg.emit_xliff_companion is True

    # Closed vocabulary validation
    with pytest.raises(ValidationError):
        TypstConfig(spill_policy="invalid_spill")  # type: ignore[arg-type]


def test_quality_config_standalone_validation() -> None:
    quality_cfg = QualityConfig(
        qe_threshold=0.85,
        max_repair_rounds=3,
        consistency_enforce="repair",
        consistency_max_repairs=30,
    )
    assert quality_cfg.qe_threshold == 0.85
    assert quality_cfg.max_repair_rounds == 3
    assert quality_cfg.consistency_enforce == "repair"
    assert quality_cfg.consistency_max_repairs == 30
    assert quality_cfg.render_preflight_enabled is True

    # Bound validation: threshold must be in [0.0, 1.0]
    with pytest.raises(ValidationError):
        QualityConfig(qe_threshold=1.5)


def test_engine_config_standalone_validation() -> None:
    engine_cfg = EngineConfig(
        db_dir=Path("/custom/dbs"),
        max_concurrency=16,
        macro_chunk_size=12,
        docling_isolate_process=True,
    )
    assert engine_cfg.db_dir == Path("/custom/dbs")
    assert engine_cfg.max_concurrency == 16
    assert engine_cfg.macro_chunk_size == 12
    assert engine_cfg.docling_isolate_process is True

    # Concurrency must be > 0
    with pytest.raises(ValidationError):
        EngineConfig(max_concurrency=0)


def test_ubt_config_subdomain_property_access() -> None:
    cfg = UBTConfig(
        draft_model="claude-3-7-sonnet",
        font_family="Noto Serif CJK SC",
        qe_threshold=0.80,
        max_concurrency=12,
    )

    # Sub-domain property access
    assert isinstance(cfg.router, RouterConfig)
    assert cfg.router.draft_model == "claude-3-7-sonnet"
    assert cfg.router.rate_limit_rpm == cfg.rate_limit_rpm

    assert isinstance(cfg.typst, TypstConfig)
    assert cfg.typst.font_family == "Noto Serif CJK SC"
    assert cfg.typst.spill_policy == cfg.spill_policy

    assert isinstance(cfg.quality, QualityConfig)
    assert cfg.quality.qe_threshold == 0.80
    assert cfg.quality.max_repair_rounds == cfg.max_repair_rounds

    assert isinstance(cfg.engine, EngineConfig)
    assert cfg.engine.max_concurrency == 12
    assert cfg.engine.db_dir == cfg.db_dir


def test_ubt_config_constructor_unpacks_subdomains() -> None:
    router_in = RouterConfig(draft_model="deepseek-v3", rate_limit_rpm=150)
    typst_in = TypstConfig(font_family="Noto Sans CJK SC", layout_columns=1)
    quality_in = QualityConfig(qe_threshold=0.88, max_repair_rounds=4)
    engine_in = EngineConfig(max_concurrency=8, db_dir=Path("/tmp/test_ledgers"))

    cfg = UBTConfig(
        router=router_in,
        typst=typst_in,
        quality=quality_in,
        engine=engine_in,
    )

    # Flat access remains 100% compatible
    assert cfg.draft_model == "deepseek-v3"
    assert cfg.rate_limit_rpm == 150
    assert cfg.font_family == "Noto Sans CJK SC"
    assert cfg.layout_columns == 1
    assert cfg.qe_threshold == 0.88
    assert cfg.max_repair_rounds == 4
    assert cfg.max_concurrency == 8
    assert cfg.db_dir == Path("/tmp/test_ledgers")

    # Re-reading properties reflects unpacked values
    assert cfg.router.draft_model == "deepseek-v3"
    assert cfg.typst.font_family == "Noto Sans CJK SC"
    assert cfg.quality.qe_threshold == 0.88
    assert cfg.engine.max_concurrency == 8


def test_as_subdomains_and_from_subdomains_roundtrip() -> None:
    original = UBTConfig(
        draft_model="gemini-2.5-pro",
        font_family="Noto Serif CJK SC",
        qe_threshold=0.82,
        max_concurrency=14,
    )

    r, t, q, e = original.as_subdomains()
    assert isinstance(r, RouterConfig)
    assert isinstance(t, TypstConfig)
    assert isinstance(q, QualityConfig)
    assert isinstance(e, EngineConfig)

    recomposed = UBTConfig.from_subdomains(router=r, typst=t, quality=q, engine=e)
    assert recomposed.draft_model == original.draft_model
    assert recomposed.font_family == original.font_family
    assert recomposed.qe_threshold == original.qe_threshold
    assert recomposed.max_concurrency == original.max_concurrency


def test_model_copy_supports_subdomain_updates() -> None:
    base = UBTConfig(
        draft_model="base-model",
        font_family="Base Font",
        qe_threshold=0.75,
    )

    # Update via typed RouterConfig
    updated = base.model_copy(update={"router": RouterConfig(draft_model="copied-model")})
    assert updated.draft_model == "copied-model"
    assert updated.router.draft_model == "copied-model"
    assert updated.font_family == "Base Font"

    # Flat update still works as before
    updated_flat = base.model_copy(update={"font_family": "New Font"})
    assert updated_flat.font_family == "New Font"
    assert updated_flat.typst.font_family == "New Font"
    assert updated_flat.draft_model == "base-model"
