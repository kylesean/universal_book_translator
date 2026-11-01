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
    payload = boxes_to_provenance((_box(3, (1.0, 2.0, 3.0, 4.0)), _box(4, (5.0, 6.0, 7.0, 8.0))))
    assert payload == [
        {"page": 3, "bbox": [1.0, 2.0, 3.0, 4.0]},
        {"page": 4, "bbox": [5.0, 6.0, 7.0, 8.0]},
    ]
    import json

    assert json.loads(json.dumps(payload)) == payload


def test_a_single_box_chain_is_not_worth_restoring() -> None:
    assert boxes_from_provenance(boxes_to_provenance((_box(1, (0.0, 0.0, 1.0, 1.0)),))) == ()


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "not a list",
        [],
        [{"page": 1}],
        [{"bbox": [1.0, 2.0, 3.0, 4.0]}],
        [{"page": 1, "bbox": [1.0, 2.0, 3.0]}],
        [{"page": "x", "bbox": [1.0, 2.0, 3.0, 4.0]}],
        [{"page": 1, "bbox": [1.0, 2.0, 3.0, 4.0]}, "junk"],
    ],
)
def test_a_malformed_chain_reads_as_empty(raw: object) -> None:
    assert boxes_from_provenance(raw) == ()
