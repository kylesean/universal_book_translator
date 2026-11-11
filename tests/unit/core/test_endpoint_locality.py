"""The machine-boundary predicate: container aliases are no longer local by name.

F5 regression: ``host.docker.internal`` and ``0.0.0.0`` were hardcoded into the
local set, so inside the shipped container the "a book page may never leave
this machine" invariant was false — those names reach the *host*, which is
another machine. Locality by name now requires the operator to list the host
in ``UBT_LOCAL_HOSTNAMES``, and the resolution must agree.
"""

from __future__ import annotations

import pytest

from ubt.core.endpoint_locality import (
    endpoint_on_this_machine,
    operator_local_hostnames,
    resolved_addresses_are_local,
)

pytestmark = pytest.mark.fast


@pytest.mark.parametrize(
    "endpoint",
    [
        None,
        "http://127.0.0.1:8765",
        "http://localhost:8765",
        "127.0.0.1:8765",
        "localhost",
        "http://[::1]:8765",
        "http://docs.example.localhost:8765",
    ],
)
def test_loopback_endpoints_stay_local(endpoint: str | None) -> None:
    assert endpoint_on_this_machine(endpoint) is True


@pytest.mark.parametrize(
    "endpoint",
    [
        # The container-host aliases reach the HOST from inside the container.
        "http://host.docker.internal:8765",
        "http://host.containers.internal:8765",
        # The unspecified address as an operator writes a bind — non-local as a
        # destination now; a local endpoint is pointed at localhost instead.
        "http://0.0.0.0:8765",
        "0.0.0.0:8765",
        # A private LAN box is another machine.
        "http://192.168.1.10:8765",
        "http://10.0.0.5:8765",
        # A public endpoint, obviously.
        "https://api.openai.com/v1",
    ],
)
def test_non_machine_endpoints_are_not_local(endpoint: str) -> None:
    assert endpoint_on_this_machine(endpoint) is False


def test_an_operator_opt_in_requires_resolution_to_agree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UBT_LOCAL_HOSTNAMES", "host.docker.internal")
    assert operator_local_hostnames() == {"host.docker.internal"}
    # "localhost" resolves to loopback, so the opt-in path accepts it...
    monkeypatch.setenv("UBT_LOCAL_HOSTNAMES", "localhost")
    assert endpoint_on_this_machine("http://localhost:8765") is True
    # ...while a name that does not resolve fails closed even when opted in.
    monkeypatch.setenv("UBT_LOCAL_HOSTNAMES", "no-such-host.invalid")
    assert endpoint_on_this_machine("http://no-such-host.invalid:8765") is False


def test_resolution_of_loopback_names_is_local() -> None:
    assert resolved_addresses_are_local("localhost") is True


def test_resolution_of_unknown_names_fails_closed() -> None:
    assert resolved_addresses_are_local("no-such-host.invalid") is False


def test_the_opt_in_never_widens_literal_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A private LAN IP stays non-local even when someone lists it."""
    monkeypatch.setenv("UBT_LOCAL_HOSTNAMES", "192.168.1.10")
    assert endpoint_on_this_machine("http://192.168.1.10:8765") is False


def test_the_default_opt_in_list_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UBT_LOCAL_HOSTNAMES", raising=False)
    assert operator_local_hostnames() == frozenset()


def test_the_registry_gate_names_the_opt_in_knob() -> None:
    from ubt.adapters.pdf.vlm.registry import assert_page_egress_allowed

    with pytest.raises(ValueError, match="UBT_LOCAL_HOSTNAMES"):
        assert_page_egress_allowed(
            "http://host.docker.internal:8765", allow_page_upload=False, mode="cloud"
        )
    # The operator opt-in or the explicit upload opt-in both silence the gate.
    assert_page_egress_allowed(
        "http://host.docker.internal:8765", allow_page_upload=True, mode="cloud"
    )


def test_the_registry_gate_still_passes_a_loopback_endpoint() -> None:
    from ubt.adapters.pdf.vlm.registry import assert_page_egress_allowed

    assert_page_egress_allowed("http://127.0.0.1:8765", allow_page_upload=False, mode="cloud")
