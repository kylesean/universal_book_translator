"""Canonical JobSpec shared across CLI, REST API, and MCP surfaces."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ubt.codecs.protocol import DeliveryMode
from ubt.core.job_options import job_id_is_valid, lang_pair_validation_error


class JobSpec(BaseModel):
    """Single source of truth for submitting or re-exporting a translation job."""

    model_config = ConfigDict(extra="forbid")

    source_path: Path
    output_path: Path | None = None
    job_id: str | None = None
    source_lang: str = "en"
    target_lang: str = "zh"
    delivery_mode: DeliveryMode = DeliveryMode.MONO
    pages: list[int] | None = None
    provider: str | None = None
    repair_provider: str | None = None
    profile: str = "auto"
    glossary: Path | None = None
    allow_overflow_compaction: bool = True
    allow_needs_human: bool = True
    dry_run: bool = False
    fresh: bool = False
    max_overflow_compaction_units: int = Field(default=32, ge=0, le=200)

    @field_validator("job_id")
    @classmethod
    def _validate_job_id(cls, value: str | None) -> str | None:
        if value is not None and not job_id_is_valid(value):
            raise ValueError(f"Invalid or reserved job_id: {value!r}")
        return value

    @field_validator("target_lang")
    @classmethod
    def _validate_langs(cls, value: str, info: object) -> str:
        data = getattr(info, "data", {}) or {}
        src = data.get("source_lang", "en")
        err = lang_pair_validation_error(source_lang=src, target_lang=value)
        if err:
            raise ValueError(err)
        return value

    @property
    def page_set(self) -> set[int] | None:
        if not self.pages:
            return None
        return {p for p in self.pages if p >= 1}

    def resolved_output_path(self) -> Path:
        if self.output_path is not None:
            return self.output_path
        stem = self.source_path.stem
        suffix = self.source_path.suffix or ".pdf"
        mode_tag = self.delivery_mode.value
        page_tag = ""
        if self.pages:
            ps = sorted(set(self.pages))
            page_tag = f"_p{ps[0]}-{ps[-1]}" if len(ps) > 1 else f"_p{ps[0]}"
        return self.source_path.with_name(f"{stem}_{self.target_lang}_{mode_tag}{page_tag}{suffix}")
