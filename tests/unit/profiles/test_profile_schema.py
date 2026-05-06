"""Unit tests — ProfileSpec validation and expansion."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.profiles.schema import (
    CustomerBoundary,
    DecoderPath,
    ProfileSpec,
    ProvenanceSource,
    RuntimeBudget,
)


def _minimal_kwargs() -> dict:
    return dict(
        profile_id="test_profile",
        name="Test profile",
        description="x" * 30,
        architecture="generic",
        intended_use="x" * 30,
        limitations="x" * 30,
        distances=(3,),
        rounds_by_distance={3: (3,)},
        bases=("X",),
        p_errors=(0.001,),
        decoder_paths=(
            DecoderPath(label="A", backend="pymatching_baseline"),
            DecoderPath(label="B", backend="pymatching_correlated"),
        ),
        boundary=CustomerBoundary(
            public_proxy_can_conclude=("public ok",),
            requires_customer_private_inputs=("customer private",),
        ),
        provenance=(
            ProvenanceSource(
                label="S1",
                url="https://github.com/quantumlib/Stim",
                cites=("distances",),
            ),
            ProvenanceSource(
                label="S2",
                url="https://pymatching.readthedocs.io/en/latest/",
                cites=("decoder_paths",),
            ),
        ),
    )


def test_profile_builds_from_minimal_kwargs() -> None:
    p = ProfileSpec(**_minimal_kwargs())
    assert p.profile_id == "test_profile"


def test_profile_rejects_even_distance() -> None:
    kw = _minimal_kwargs()
    kw["distances"] = (4,)
    kw["rounds_by_distance"] = {4: (3,)}
    with pytest.raises(ValidationError):
        ProfileSpec(**kw)


def test_profile_rejects_p_error_out_of_range() -> None:
    kw = _minimal_kwargs()
    kw["p_errors"] = (0.0,)
    with pytest.raises(ValidationError):
        ProfileSpec(**kw)
    kw["p_errors"] = (0.5,)
    with pytest.raises(ValidationError):
        ProfileSpec(**kw)


def test_profile_rejects_rounds_missing_for_declared_distance() -> None:
    kw = _minimal_kwargs()
    kw["distances"] = (3, 5)
    kw["rounds_by_distance"] = {3: (3,)}  # 5 missing
    with pytest.raises(ValidationError):
        ProfileSpec(**kw)


def test_profile_rejects_fewer_than_2_decoder_paths() -> None:
    kw = _minimal_kwargs()
    kw["decoder_paths"] = (DecoderPath(label="Only", backend="pymatching_baseline"),)
    with pytest.raises(ValidationError):
        ProfileSpec(**kw)


def test_profile_rejects_placeholder_url() -> None:
    with pytest.raises(ValidationError):
        ProvenanceSource(
            label="bad",
            url="https://example.com/x",
            cites=("distances",),
        )


def test_profile_rejects_missing_customer_boundary() -> None:
    with pytest.raises(ValidationError):
        CustomerBoundary(
            public_proxy_can_conclude=(),
            requires_customer_private_inputs=("x",),
        )
    with pytest.raises(ValidationError):
        CustomerBoundary(
            public_proxy_can_conclude=("x",),
            requires_customer_private_inputs=(),
        )


def test_runtime_budget_orders_target_le_cap() -> None:
    with pytest.raises(ValidationError):
        RuntimeBudget(latency_us_target=50.0, latency_us_hard_cap=20.0)


def test_expand_points_is_deterministic_and_sorted() -> None:
    p = ProfileSpec(**_minimal_kwargs())
    a = p.expand_points()
    b = p.expand_points()
    assert a == b
    # Sorted by (distance, rounds, basis, p_error).
    for i in range(1, len(a)):
        assert (a[i]["distance"], a[i]["rounds"], a[i]["basis"], a[i]["p_error"]) >= (
            a[i - 1]["distance"],
            a[i - 1]["rounds"],
            a[i - 1]["basis"],
            a[i - 1]["p_error"],
        )


def test_profile_id_pattern_enforced() -> None:
    kw = _minimal_kwargs()
    kw["profile_id"] = "NOT-lower-snake"
    with pytest.raises(ValidationError):
        ProfileSpec(**kw)


def test_frozen_profile_rejects_mutation() -> None:
    p = ProfileSpec(**_minimal_kwargs())
    with pytest.raises(ValidationError):
        p.profile_id = "other"  # type: ignore[misc]
