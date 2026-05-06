"""Tests for :mod:`app.core.capability_report`.

The schema is the contract between decoder adapters and the deployment-
readiness report. Most of the risk lives at the boundary: wrong
``blocker_category`` values, mutation after construction, or missing
required fields would all corrupt the Risk Register without raising
anywhere obvious. Tests therefore concentrate on adversarial inputs —
unknown literal values, extra fields, mutation attempts, missing
required fields — rather than just exercising the happy path.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.capability_report import CapabilityReport


def test_unavailable_sets_blocker_category() -> None:
    report = CapabilityReport.unavailable(
        reason="tensorrt python module not importable",
        required=["tensorrt-cu13>=10.16"],
        category="not_installed",
    )

    assert report.available is False
    assert report.blocker_category == "not_installed"
    assert report.reason == "tensorrt python module not importable"
    assert report.required == ["tensorrt-cu13>=10.16"]
    assert report.detected_versions == {}
    assert report.probe_latency_ms is None


def test_unavailable_rejects_none_category() -> None:
    # 'none' is reserved for available capabilities; using it for an
    # unavailable report would silently drop the entry from the Risk
    # Register. The factory must refuse it.
    with pytest.raises(ValueError, match="reserved for available"):
        CapabilityReport.unavailable(
            reason="x",
            required=["pkg"],
            category="none",
        )


def test_ready_requires_non_none_blocker() -> None:
    report = CapabilityReport.ready(
        reason="torch 2.11.0+cu130 imported",
        required=["torch"],
        detected_versions={"torch": "2.11.0+cu130"},
        probe_latency_ms=12.5,
    )

    # "Non-None blocker" here means the field is always populated with a
    # Literal value; the schema does not allow None even on the ready
    # path. 'none' is the sentinel for "nothing is blocking".
    assert report.blocker_category is not None
    assert report.blocker_category == "none"
    assert report.available is True
    assert report.probe_latency_ms == 12.5
    assert report.detected_versions == {"torch": "2.11.0+cu130"}


def test_ready_rejects_explicit_none_for_blocker_category() -> None:
    # Direct construction must also refuse None — the Literal type
    # enforces this regardless of which classmethod is used.
    with pytest.raises(ValidationError):
        CapabilityReport(
            available=True,
            reason="ok",
            required=["torch"],
            blocker_category=None,  # type: ignore[arg-type]
        )


def test_model_is_frozen_and_forbids_extra() -> None:
    report = CapabilityReport.ready(
        reason="ok",
        required=["torch"],
        detected_versions={"torch": "2.11.0+cu130"},
    )

    # model_config exposes the frozen flag used by downstream code that
    # introspects the schema (e.g. documentation generators).
    assert report.model_config.get("frozen") is True
    assert report.model_config.get("extra") == "forbid"

    with pytest.raises(ValidationError):
        report.available = False
    with pytest.raises(ValidationError):
        report.reason = "changed"

    # Extra fields at construction time must be rejected; silent
    # acceptance would let typos like 'block_category' slip through and
    # vanish from the Risk Register join.
    with pytest.raises(ValidationError):
        CapabilityReport(
            available=True,
            reason="ok",
            required=["torch"],
            blocker_category="none",
            surprise_field="nope",  # type: ignore[call-arg]
        )


def test_blocker_category_literal_rejects_unknown_value() -> None:
    # A free-form string would silently accept typos. The Literal must
    # reject anything outside the documented seven-value set.
    with pytest.raises(ValidationError):
        CapabilityReport(
            available=False,
            reason="broken",
            required=["pkg"],
            blocker_category="mystery",  # type: ignore[arg-type]
        )

    # Sanity-check every documented value is accepted so the Literal
    # does not drift out of sync with the Risk Register buckets.
    for value in (
        "none",
        "machine",
        "software",
        "licensing",
        "runtime",
        "not_installed",
        "version_mismatch",
    ):
        accepted = CapabilityReport(
            available=(value == "none"),
            reason="check",
            required=["pkg"],
            blocker_category=value,
        )
        assert accepted.blocker_category == value


def test_detected_versions_defaults_to_empty_dict() -> None:
    report = CapabilityReport.unavailable(
        reason="cuquantum not importable",
        required=["cuquantum-python-cu13>=26.3"],
        category="not_installed",
    )

    assert report.detected_versions == {}

    # Two unavailable reports must not share the same underlying dict —
    # a mutable default would let the Risk Register entries for
    # different capabilities silently alias each other.
    other = CapabilityReport.unavailable(
        reason="cudaq not importable",
        required=["cudaq>=0.14"],
        category="not_installed",
    )
    assert report.detected_versions is not other.detected_versions


def test_required_field_non_optional() -> None:
    # 'required' must be supplied; the schema rejects omission so that
    # downstream reports never see an ambiguous "capability verified
    # against nothing" record.
    with pytest.raises(ValidationError):
        CapabilityReport(  # type: ignore[call-arg]
            available=False,
            reason="broken",
            blocker_category="not_installed",
        )

    # An empty list is also rejected — "checked zero preconditions" is
    # meaningless in the Risk Register.
    with pytest.raises(ValidationError):
        CapabilityReport(
            available=False,
            reason="broken",
            required=[],
            blocker_category="not_installed",
        )


def test_reason_must_be_non_empty() -> None:
    # An empty reason would render as a blank line in the deployment-
    # readiness report. Reject it at the boundary.
    with pytest.raises(ValidationError):
        CapabilityReport(
            available=False,
            reason="",
            required=["pkg"],
            blocker_category="not_installed",
        )
