"""The box-chain provenance round-trip: write and read must agree.

An element spanning several physical boxes (a paragraph crossing a page break)
serializes its chain into ``block.provenance.physical_boxes`` for the ledger,
which cannot store the typed ``CompositeSpan``. The shape was hand-rolled in
three writers and one reader; a typo in any of them silently collapsed the chain
to the element's first box on reload, squeezing a whole translation into one
line. These pin the single shared pair.
"""

from __future__ import annotations

import pytest

from ubt.model.span import (
    PhysicalBox,
    boxes_from_provenance,
    boxes_to_provenance,
)

pytestmark = pytest.mark.fast


def _box(page: int, bbox: tuple[float, float, float, float]) -> PhysicalBox:
    return PhysicalBox.of(page, bbox)


def test_a_chain_round_trips_through_provenance() -> None:
    boxes = (
        _box(1, (10.0, 20.0, 210.0, 45.0)),
        _box(2, (10.0, 50.0, 210.0, 75.0)),
    )
    rebuilt = boxes_from_provenance(boxes_to_provenance(boxes))
    assert rebuilt == boxes


def test_the_serialized_form_is_json_shaped() -> None:
    # The ledger stores it as JSON; bbox must be a plain list, not a tuple.
    payload = boxes_to_provenance((_box(3, (1.0, 2.0, 3.0, 4.0)), _box(4, (5.0, 6.0, 7.0, 8.0))))
    assert payload == [
        {"page": 3, "bbox": [1.0, 2.0, 3.0, 4.0]},
        {"page": 4, "bbox": [5.0, 6.0, 7.0, 8.0]},
    ]
    import json

    assert json.loads(json.dumps(payload)) == payload


def test_a_single_box_chain_is_not_worth_restoring() -> None:
    # One box adds nothing over the element's own Span, so the reader returns
    # empty and the caller keeps the element's span.
    assert boxes_from_provenance(boxes_to_provenance((_box(1, (0.0, 0.0, 1.0, 1.0)),))) == ()


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "not a list",
        [],
        [{"page": 1}],  # missing bbox
        [{"bbox": [1.0, 2.0, 3.0, 4.0]}],  # missing page
        [{"page": 1, "bbox": [1.0, 2.0, 3.0]}],  # short bbox
        [{"page": "x", "bbox": [1.0, 2.0, 3.0, 4.0]}],  # non-int page
        [{"page": 1, "bbox": [1.0, 2.0, 3.0, 4.0]}, "junk"],  # one bad entry
    ],
)
def test_a_malformed_chain_reads_as_empty(raw: object) -> None:
    # Any malformed entry voids the whole chain: a partial chain would misplace
    # the tail of the translation rather than fall back to the element's span.
    assert boxes_from_provenance(raw) == ()
