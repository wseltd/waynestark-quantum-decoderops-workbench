"""TensorRT engine metadata sidecar (T053)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = ["skip_reason", "write_tensorrt_engine_meta"]


def _sha256_stream(path: Path, chunk: int = 64 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_tensorrt_engine_meta(
    engine_path: Path,
    adapter_metadata: dict[str, Any],
    output_dir: Path,
) -> Path:
    engine_path = Path(engine_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine_sha256 = _sha256_stream(engine_path)
    payload = {
        "engine_path": engine_path.name,
        "engine_sha256": engine_sha256,
        "trt_version": adapter_metadata.get("trt_version"),
        "cuda_version": adapter_metadata.get("cuda_version"),
        "builder_flags": adapter_metadata.get("builder_flags", []),
        "input_shapes": adapter_metadata.get("input_shapes", {}),
        "output_shapes": adapter_metadata.get("output_shapes", {}),
        "precision_mode": adapter_metadata.get("precision_mode"),
        "build_timestamp_utc": adapter_metadata.get("build_timestamp_utc"),
        "source_onnx_sha256": adapter_metadata.get("source_onnx_sha256"),
    }
    out = output_dir / "engine_meta.json"
    out.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return out


def skip_reason(adapter_metadata: dict[str, Any]) -> str:
    if adapter_metadata.get("available") is True:
        return "adapter reports available — not skipped"
    reason = adapter_metadata.get("reason")
    if not reason:
        return "tensorrt adapter unavailable (no reason provided)"
    return str(reason)
