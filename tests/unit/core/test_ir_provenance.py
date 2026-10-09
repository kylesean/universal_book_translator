"""Unit tests for strongly typed BlockProvenance model and IRBlock integration."""

from __future__ import annotations

import json
from collections.abc import Mapping, MutableMapping

import pytest

from ubt.core.ir.models import BlockProvenance, BlockType, IRBlock, make_element

pytestmark = pytest.mark.fast


def test_block_provenance_field_typing() -> None:
    prov = BlockProvenance(
        source_page=3,
        page_kind="cover",
        is_bold=True,
        is_italic=False,
        font_size=12.5,
        toc_entry=True,
        toc_page="42",
        fused_sources=["Part A", "Part B"],
        fused_block_ids=["b1", "b2"],
        bifurcated_from="b_orig",
        bifurcation_index=1,
    )
    assert prov.source_page == 3
    assert prov.page_kind == "cover"
    assert prov.is_bold is True
    assert prov.is_italic is False
    assert prov.font_size == 12.5
    assert prov.toc_entry is True
    assert prov.toc_page == "42"
    assert prov.fused_sources == ["Part A", "Part B"]
    assert prov.fused_block_ids == ["b1", "b2"]
    assert prov.bifurcated_from == "b_orig"
    assert prov.bifurcation_index == 1


def test_block_provenance_mapping_protocol() -> None:
    prov = BlockProvenance(is_bold=True, font_size=14.0)

    # Virtual subclass of Mapping / MutableMapping
    assert isinstance(prov, Mapping)
    assert isinstance(prov, MutableMapping)

    # Bracket indexing
    assert prov["is_bold"] is True
    assert prov["font_size"] == 14.0
    assert prov["toc_entry"] is None

    # Membership
    assert "is_bold" in prov
    assert "font_size" in prov
    assert "toc_entry" not in prov
    assert "nonexistent" not in prov

    # .get()
    assert prov.get("is_bold") is True
    assert prov.get("toc_entry") is None
    assert prov.get("nonexistent", "fallback") == "fallback"

    # Mutation via bracket
    prov["is_italic"] = True
    assert prov.is_italic is True
    assert "is_italic" in prov

    # Deletion
    del prov["is_italic"]
    assert prov.is_italic is None
    assert "is_italic" not in prov


def test_block_provenance_extra_fields() -> None:
    prov = BlockProvenance.model_validate({"custom_prop": "value_1"})
    assert prov["custom_prop"] == "value_1"
    assert "custom_prop" in prov
    assert prov.get("custom_prop") == "value_1"

    prov["dynamic_prop"] = 999
    assert prov["dynamic_prop"] == 999
    assert prov.to_dict()["dynamic_prop"] == 999

    del prov["dynamic_prop"]
    assert "dynamic_prop" not in prov


def test_block_provenance_dict_unpacking_and_equality() -> None:
    prov = BlockProvenance(is_bold=True, font_size=11.0)
    prov["extra_key"] = "extra_val"

    # Dict unpacking
    unpacked = {**prov, "added": 1}
    assert unpacked == {
        "is_bold": True,
        "font_size": 11.0,
        "extra_key": "extra_val",
        "added": 1,
    }

    # Equality with dict
    assert prov == {
        "is_bold": True,
        "font_size": 11.0,
        "extra_key": "extra_val",
    }


def test_ir_block_provenance_coercion() -> None:
    el = make_element(id="b1", spine_index=0, block_type=BlockType.NARRATIVE, source_text="test")

    # Initialized with dict (coerced at runtime via field_validator)
    block1 = IRBlock(element=el, provenance={"is_bold": True, "source_page": 2})  # type: ignore[arg-type]
    assert isinstance(block1.provenance, BlockProvenance)
    assert block1.provenance.is_bold is True
    assert block1.provenance.source_page == 2

    # Assignment with dict
    block1.provenance = {"toc_entry": True, "toc_page": "5"}  # type: ignore[assignment]
    assert isinstance(block1.provenance, BlockProvenance)
    assert block1.provenance.toc_entry is True
    assert block1.provenance.toc_page == "5"

    # Assignment with None falls back to empty BlockProvenance
    block1.provenance = None  # type: ignore[assignment]
    assert isinstance(block1.provenance, BlockProvenance)
    assert block1.provenance.to_dict() == {}


def test_block_provenance_json_roundtrip() -> None:
    prov = BlockProvenance(
        physical_boxes=[{"page": 1, "bbox": [10.0, 20.0, 30.0, 40.0]}],
        is_bold=True,
    )
    serialized = json.dumps(prov.to_dict())
    deserialized_data = json.loads(serialized)
    restored = BlockProvenance(**deserialized_data)
    assert restored.is_bold is True
    assert len(restored.physical_boxes) == 1
    assert restored.physical_boxes[0]["page"] == 1
