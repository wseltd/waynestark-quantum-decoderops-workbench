"""Schema-versioned manifest writer for artefact tarballs (T048)."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, field_validator

from app.core.fingerprint import ReproducibilityFingerprint

__all__ = [
    "MANIFEST_FILENAME",
    "ArtefactEntry",
    "Manifest",
    "build_manifest",
    "load_manifest",
    "write_manifest",
]

MANIFEST_FILENAME: str = "manifest.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

_REQUIRED_FINGERPRINT_KEYS = (
    "git_sha",
    "pip_freeze_digest",
    "config_hash",
    "rng_master_seed",
    "python_version",
    "os",
    "cpu_model",
    "cpu_count",
    "gpu_model",
    "gpu_count",
    "gpu_driver_version",
    "cuda_runtime_version",
    "timestamp_utc",
)

_ArtefactKind = Literal[
    "onnx",
    "tensorrt_engine",
    "cudaq_qec_bin",
    "parquet",
    "report_md",
    "report_html",
    "report_pdf",
    "report_json",
    "config_snapshot",
    "log",
    "other",
]


class ArtefactEntry(BaseModel):
    path: str
    sha256: str
    size_bytes: int
    kind: _ArtefactKind

    @field_validator("sha256")
    @classmethod
    def _valid_sha256(cls, v: str) -> str:
        if not _SHA256_RE.match(v):
            raise ValueError(f"invalid sha256: {v!r}")
        return v

    @field_validator("size_bytes")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"size_bytes must be >= 0; got {v}")
        return v

    @field_validator("path")
    @classmethod
    def _relative_path(cls, v: str) -> str:
        if not v:
            raise ValueError("path must be non-empty")
        if v.startswith("/"):
            raise ValueError(f"path must be relative; got {v!r}")
        parts = v.split("/")
        if ".." in parts:
            raise ValueError(f"path must not contain '..'; got {v!r}")
        return v


class Manifest(BaseModel):
    schema_version: Literal["1"] = "1"
    run_id: str
    created_at_utc: str
    config_hash: str
    fingerprint: dict[str, Any]
    artefacts: list[ArtefactEntry]

    @field_validator("run_id")
    @classmethod
    def _non_empty_run_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("run_id must be non-empty")
        return v

    @field_validator("config_hash")
    @classmethod
    def _valid_sha256(cls, v: str) -> str:
        if not _SHA256_RE.match(v):
            raise ValueError(f"invalid config_hash: {v!r}")
        return v

    @field_validator("created_at_utc")
    @classmethod
    def _iso_zulu(cls, v: str) -> str:
        if not _ISO_RE.match(v):
            raise ValueError(f"created_at_utc must be ISO 8601 Z; got {v!r}")
        return v


def _fingerprint_to_dict(fp: Any) -> dict[str, Any]:
    """Normalise a T011 ReproducibilityFingerprint OR a dict into a flat dict.

    The Manifest's fingerprint field uses a broader key set than T011's native
    field names; map T011 field names to the manifest key names.
    """
    if isinstance(fp, ReproducibilityFingerprint):
        d = fp.model_dump(mode="json")
        # Map T011 schema -> manifest keys
        mapped: dict[str, Any] = {
            "git_sha": d["git_sha"],
            "pip_freeze_digest": d["pip_freeze_digest"],
            "config_hash": d["config_hash"],
            "rng_master_seed": d["master_seed"],
            "python_version": d["python_version"],
            "os": f"{d['os_name']} {d['os_kernel']}".strip(),
            "cpu_model": d["cpu_model"],
            "cpu_count": d["cpu_count"],
            "gpu_model": ",".join(d["gpu_models"]) if d["gpu_models"] else "",
            "gpu_count": d["gpu_count"],
            "gpu_driver_version": d["nvidia_driver_version"] or "",
            "cuda_runtime_version": d["cuda_runtime_version"] or "",
            "timestamp_utc": d["timestamp_utc"],
        }
        return mapped
    if isinstance(fp, dict):
        missing = [k for k in _REQUIRED_FINGERPRINT_KEYS if k not in fp]
        if missing:
            raise ValueError(
                f"fingerprint dict missing required keys: {missing}"
            )
        return dict(fp)
    raise TypeError(
        f"fingerprint must be ReproducibilityFingerprint or dict; "
        f"got {type(fp).__name__}"
    )


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_manifest(
    run_id: str,
    fingerprint: Any,
    artefacts: list[ArtefactEntry],
    config_hash: str,
    created_at_utc: str | None = None,
) -> Manifest:
    fp_dict = _fingerprint_to_dict(fingerprint)
    # Final check: all required manifest keys present.
    missing = [k for k in _REQUIRED_FINGERPRINT_KEYS if k not in fp_dict]
    if missing:
        raise ValueError(
            f"fingerprint missing required manifest keys: {missing}"
        )
    return Manifest(
        schema_version="1",
        run_id=run_id,
        created_at_utc=created_at_utc or _now_utc_iso(),
        config_hash=config_hash,
        fingerprint=fp_dict,
        artefacts=artefacts,
    )


def write_manifest(manifest: Manifest, output_path: str | os.PathLike) -> str:
    payload = json.dumps(
        manifest.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload + "\n", encoding="utf-8")
    return str(out)


def load_manifest(path: str | os.PathLike) -> Manifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Manifest.model_validate(data)
