"""Canonical process configuration for DecoderOps.

This module defines the single source of truth for runtime settings. Every
other package should call :func:`get_settings` rather than reading environment
variables directly. Keeping this centralised means env-var names, defaults,
and validation rules live in one place and cannot drift between modules.

Design notes (what we deliberately chose):

* ``env_file=None`` — we do not auto-load ``.env``. Explicit-over-implicit
  avoids surprising test runs that pick up a developer's local file.
* ``get_settings`` is an ``lru_cache``'d accessor; there is no module-level
  ``settings = Settings()`` instance. That would freeze environment state at
  import time and be impossible to reset cleanly in tests.
* No filesystem side effects on import. Directory creation is the caller's
  job (e.g. a bootstrap or CLI command), not configuration's.
* YAML / JSON overlay parsing is explicitly out of scope — T007 owns it.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Accepted values for ``log_level``. Defined once here so the validator and
# any future caller (e.g. logging bootstrap) cannot drift apart.
VALID_LOG_LEVELS: frozenset[str] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

# Default artefact/database locations sit under ``.decoderops/`` so a clean
# checkout is runnable without any environment configuration. DuckDB gives us
# a zero-install local persistence story; Postgres is opt-in via env override.
DEFAULT_DATABASE_URL = "duckdb:///./.decoderops/decoderops.duckdb"
DEFAULT_ARTEFACT_DIR = Path("./.decoderops/artefacts")
DEFAULT_VENDOR_ISING_DIR = Path("./vendor/Ising-Decoding")
DEFAULT_ENV_REPORT_PATH = Path("./.decoderops/environment_report.json")


class Settings(BaseSettings):
    """Process-wide configuration loaded from environment variables.

    All fields are populated from ``DECODEROPS_*`` environment variables. The
    defaults are chosen so the product is runnable from a clean checkout with
    zero configuration; production deployments override via environment.

    Attributes:
        database_url: SQLAlchemy-style URL. Defaults to a local DuckDB file.
        artefact_dir: Directory where packaged run artefacts are written.
        seed: RNG master seed; ``0`` means "use the default deterministic
            stream". Per-worker seeds are derived from this value elsewhere.
        log_level: Root log level name. Validated against
            :data:`VALID_LOG_LEVELS`.
        vendor_ising_dir: Location of the NVIDIA Ising-Decoding checkout.
        environment_report_path: Path to the bootstrap-generated capability
            report consumed by the capability detector.
    """

    # ``DECODEROPS_DB_URL`` and ``DECODEROPS_ENV_REPORT`` are contract names
    # other components already rely on; using ``alias`` bypasses the normal
    # prefix+field-name derivation so we can keep the contract exact while the
    # field names stay Pythonic.
    database_url: str = Field(
        default=DEFAULT_DATABASE_URL,
        alias="DECODEROPS_DB_URL",
    )
    artefact_dir: Path = DEFAULT_ARTEFACT_DIR
    seed: int = 0
    log_level: str = "INFO"
    vendor_ising_dir: Path = DEFAULT_VENDOR_ISING_DIR
    environment_report_path: Path = Field(
        default=DEFAULT_ENV_REPORT_PATH,
        alias="DECODEROPS_ENV_REPORT",
    )

    model_config = SettingsConfigDict(
        env_prefix="DECODEROPS_",
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, value: object) -> str:
        """Uppercase and validate ``log_level`` against the allowed set.

        Raises:
            ValueError: If the value is not one of :data:`VALID_LOG_LEVELS`.
        """
        if not isinstance(value, str):
            raise ValueError(
                f"log_level must be a string, got {type(value).__name__}"
            )
        normalised = value.strip().upper()
        if normalised not in VALID_LOG_LEVELS:
            allowed = ", ".join(sorted(VALID_LOG_LEVELS))
            raise ValueError(
                f"log_level {value!r} is not valid; expected one of: {allowed}"
            )
        return normalised

    def __repr__(self) -> str:
        """Deterministic representation without surfacing secret-like fields.

        No current field is a credential, but keeping the repr narrow means
        adding a ``SecretStr`` later cannot accidentally leak via logging.
        """
        return (
            f"Settings(database_url={self.database_url!r}, "
            f"artefact_dir={self.artefact_dir.as_posix()!r}, "
            f"seed={self.seed}, "
            f"log_level={self.log_level!r}, "
            f"vendor_ising_dir={self.vendor_ising_dir.as_posix()!r}, "
            f"environment_report_path="
            f"{self.environment_report_path.as_posix()!r})"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton.

    The result is cached so repeated calls do not re-read environment
    variables. Tests that mutate the environment must call
    :func:`reset_settings_cache` to force a reload.
    """
    return Settings()


def reset_settings_cache() -> None:
    """Clear the :func:`get_settings` cache.

    Intended for tests that need to observe a fresh environment. Production
    code should not need this — environment variables are read once per
    process lifetime.
    """
    get_settings.cache_clear()
