"""Built-in profile registry.

Profile definition modules under ``app.profiles.defs`` each export a
module-level ``PROFILE: ProfileSpec`` constant. The registry imports
them eagerly so profile_id uniqueness is enforced at package load.
"""

from __future__ import annotations

from collections.abc import Iterable

from app.profiles.defs import (
    ai_predecoder_export_runtime,
    generic_surface_code_readiness,
    superconducting_latency_aware,
    trapped_ion_looser_latency,
)
from app.profiles.schema import ProfileSpec

__all__ = [
    "PROFILES",
    "available_profile_ids",
    "get_profile",
    "iter_profiles",
]


_BUILTIN: tuple[ProfileSpec, ...] = (
    generic_surface_code_readiness.PROFILE,
    superconducting_latency_aware.PROFILE,
    ai_predecoder_export_runtime.PROFILE,
    trapped_ion_looser_latency.PROFILE,
)


def _build_registry(specs: Iterable[ProfileSpec]) -> dict[str, ProfileSpec]:
    out: dict[str, ProfileSpec] = {}
    for s in specs:
        if s.profile_id in out:
            raise RuntimeError(f"duplicate profile_id in registry: {s.profile_id!r}")
        out[s.profile_id] = s
    return out


PROFILES: dict[str, ProfileSpec] = _build_registry(_BUILTIN)


class ProfileNotFoundError(KeyError):
    """Raised when a profile_id is not in the registry."""


def get_profile(profile_id: str) -> ProfileSpec:
    if profile_id not in PROFILES:
        raise ProfileNotFoundError(
            f"unknown profile_id {profile_id!r}; known ids: {sorted(PROFILES)}"
        )
    return PROFILES[profile_id]


def available_profile_ids() -> list[str]:
    return sorted(PROFILES)


def iter_profiles() -> list[ProfileSpec]:
    return [PROFILES[pid] for pid in available_profile_ids()]
