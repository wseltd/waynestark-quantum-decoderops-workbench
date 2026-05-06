"""Unit tests for :mod:`app.config.settings`.

These tests lock down the contract other modules will depend on:
env-prefix naming, default values, type coercion, validator behaviour, and
cache semantics. Each test isolates ``DECODEROPS_*`` env vars so tests cannot
leak state into one another.
"""

from __future__ import annotations

import os
from collections.abc import Generator

import pytest
from pydantic import ValidationError

from app.config.settings import (
    DEFAULT_ARTEFACT_DIR,
    DEFAULT_DATABASE_URL,
    Settings,
    get_settings,
    reset_settings_cache,
)

# Every env var read by Settings — cleared before and after each test so a
# stray ``DECODEROPS_SEED`` in the developer's shell cannot pollute results.
DECODEROPS_ENV_VARS = (
    "DECODEROPS_DB_URL",
    "DECODEROPS_ARTEFACT_DIR",
    "DECODEROPS_SEED",
    "DECODEROPS_LOG_LEVEL",
    "DECODEROPS_VENDOR_ISING_DIR",
    "DECODEROPS_ENV_REPORT",
)


@pytest.fixture(autouse=True)
def _clean_env(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """Strip DecoderOps env vars and reset the settings cache around each test.

    Using ``monkeypatch.delenv`` with ``raising=False`` means tests run
    identically whether or not the developer's shell has any of these set.
    """
    for name in DECODEROPS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


def test_defaults_when_no_env_set() -> None:
    settings = Settings()
    assert settings.database_url == DEFAULT_DATABASE_URL
    assert settings.artefact_dir == DEFAULT_ARTEFACT_DIR
    assert settings.artefact_dir.as_posix().endswith(".decoderops/artefacts")
    assert settings.seed == 0
    assert settings.log_level == "INFO"
    assert settings.vendor_ising_dir.as_posix().endswith("vendor/Ising-Decoding")
    assert settings.environment_report_path.as_posix().endswith(
        ".decoderops/environment_report.json"
    )


def test_env_override_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "DECODEROPS_DB_URL",
        "postgresql+psycopg://u:p@localhost:5432/decoderops",
    )
    settings = Settings()
    assert settings.database_url == (
        "postgresql+psycopg://u:p@localhost:5432/decoderops"
    )


def test_env_override_seed_parsed_as_int(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DECODEROPS_SEED", "1337")
    settings = Settings()
    assert settings.seed == 1337
    assert isinstance(settings.seed, int)

    # Non-integer values must raise a ValidationError at the pydantic boundary,
    # not silently coerce or fall back to the default.
    monkeypatch.setenv("DECODEROPS_SEED", "not-a-number")
    with pytest.raises(ValidationError):
        Settings()


def test_invalid_log_level_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DECODEROPS_LOG_LEVEL", "LOUD")
    with pytest.raises(ValidationError) as excinfo:
        Settings()
    # The validator's message must name the offending value so operators can
    # fix the env var without reading source code.
    assert "LOUD" in str(excinfo.value)

    # Case-insensitive acceptance: lower-case should normalise to upper-case.
    monkeypatch.setenv("DECODEROPS_LOG_LEVEL", "debug")
    settings = Settings()
    assert settings.log_level == "DEBUG"


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DECODEROPS_SEED", "7")
    first = get_settings()
    # Mutating the environment after the first call must NOT be visible until
    # the cache is explicitly reset — that's the contract callers rely on.
    monkeypatch.setenv("DECODEROPS_SEED", "9")
    second = get_settings()
    assert first is second
    assert second.seed == 7


def test_reset_settings_cache_reloads_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DECODEROPS_SEED", "11")
    first = get_settings()
    assert first.seed == 11

    monkeypatch.setenv("DECODEROPS_SEED", "23")
    reset_settings_cache()
    second = get_settings()
    assert second is not first
    assert second.seed == 23


def test_env_prefix_is_DECODEROPS(monkeypatch: pytest.MonkeyPatch) -> None:
    # An env var without the DECODEROPS_ prefix must be ignored — this guards
    # against accidental adoption of generic names like ``SEED`` or ``DB_URL``.
    monkeypatch.setenv("SEED", "999")
    monkeypatch.setenv("DB_URL", "sqlite:///ignored.db")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    settings = Settings()
    assert settings.seed == 0
    assert settings.database_url == DEFAULT_DATABASE_URL
    assert settings.log_level == "INFO"

    # And the prefix itself is case-insensitive for the suffix portion.
    monkeypatch.setenv("decoderops_seed", "5")
    reloaded = Settings()
    assert reloaded.seed == 5

    # Sanity: the configured prefix really is DECODEROPS_.
    assert Settings.model_config["env_prefix"] == "DECODEROPS_"
    # And it's not being read from any of the unprefixed names we just set.
    assert "SEED" in os.environ and os.environ["SEED"] == "999"
