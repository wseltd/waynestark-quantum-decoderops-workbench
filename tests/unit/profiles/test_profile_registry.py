"""Unit tests — registry + loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.errors import IngestionError
from app.profiles.loader import load_profile, load_profile_from_dict
from app.profiles.registry import (
    PROFILES,
    ProfileNotFoundError,
    available_profile_ids,
    get_profile,
    iter_profiles,
)
from app.profiles.schema import ProfileSpec


def test_registry_has_at_least_three_serious_profiles() -> None:
    ids = available_profile_ids()
    # Serious public-proxy profiles per the research pack.
    required = {
        "generic_surface_code_readiness",
        "superconducting_latency_aware",
        "ai_predecoder_export_runtime",
    }
    assert required.issubset(set(ids))


def test_registry_profile_ids_are_unique() -> None:
    assert len(PROFILES) == len(available_profile_ids())


def test_get_profile_raises_on_unknown() -> None:
    with pytest.raises(ProfileNotFoundError):
        get_profile("nope_nope_nope")


def test_each_registered_profile_is_a_valid_profilespec() -> None:
    for p in iter_profiles():
        assert isinstance(p, ProfileSpec)
        assert p.boundary.public_proxy_can_conclude
        assert p.boundary.requires_customer_private_inputs
        assert len(p.decoder_paths) >= 2
        # Every profile has at least 2 primary-source references.
        assert len(p.provenance) >= 2


def test_superconducting_profile_has_runtime_budget_with_willow_envelope() -> None:
    p = get_profile("superconducting_latency_aware")
    assert p.runtime_budget is not None
    # Willow paper: 63 μs decoder latency, 1.1 μs cycle time.
    assert p.runtime_budget.latency_us_hard_cap == 63.0
    assert p.runtime_budget.cycle_time_us == 1.1
    # Riverlane Deltaflow 2: <20 μs target.
    assert p.runtime_budget.latency_us_target == 20.0


def test_ai_predecoder_profile_ships_canonical_nvidia_point() -> None:
    p = get_profile("ai_predecoder_export_runtime")
    # Canonical public example: d=13 rounds=104 basis=X p=0.003.
    assert p.distances == (13,)
    assert p.rounds_by_distance == {13: (104,)}
    assert p.bases == ("X",)
    assert p.p_errors == (0.003,)


def test_trapped_ion_profile_has_caution_label() -> None:
    p = get_profile("trapped_ion_looser_latency")
    assert p.caution_label, "trapped-ion profile must ship with a caution_label"


def test_loader_from_dict_round_trips() -> None:
    p_src = get_profile("generic_surface_code_readiness")
    d = p_src.model_dump(mode="json")
    p_round = load_profile_from_dict(d)
    assert p_round == p_src


def test_loader_from_file_yaml(tmp_path: Path) -> None:
    import yaml

    p_src = get_profile("generic_surface_code_readiness")
    d = p_src.model_dump(mode="json")
    fn = tmp_path / "p.yaml"
    fn.write_text(yaml.safe_dump(d))
    p_loaded = load_profile(fn)
    assert p_loaded == p_src


def test_loader_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(IngestionError):
        load_profile(tmp_path / "nope.yaml")


def test_loader_rejects_unsupported_suffix(tmp_path: Path) -> None:
    p = tmp_path / "p.ini"
    p.write_text("hello")
    with pytest.raises(IngestionError):
        load_profile(p)


def test_loader_rejects_malformed_profile(tmp_path: Path) -> None:
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"profile_id": "x"}))  # too-short, missing fields
    with pytest.raises(IngestionError):
        load_profile(p)
