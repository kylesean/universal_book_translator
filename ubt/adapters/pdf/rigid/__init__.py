"""Rigid typesetting: region zones plus adaptive over-region typesetting."""

from ubt.adapters.pdf.rigid.extract import extract_pages
from ubt.adapters.pdf.rigid.typesetter import (
    RigidPageReport,
    RigidReport,
    RigidTypesetter,
)
from ubt.adapters.pdf.rigid.zones import PageFacts, Zone, build_zones

__all__ = [
    "RigidPageReport",
    "RigidReport",
    "RigidTypesetter",
    "PageFacts",
    "Zone",
    "build_zones",
    "extract_pages",
]
