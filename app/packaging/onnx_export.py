"""Decoder → ONNX export with SHA256 sidecar + manifest (T052)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.packaging.sha256_stamp import stamp_file

__all__ = [
    "DEFAULT_OPSET",
    "OnnxExportResult",
    "OnnxInputSignature",
    "export_decoder_to_onnx",
]

DEFAULT_OPSET: int = 17
_LOG = logging.getLogger(__name__)


class OnnxInputSignature(BaseModel):
    name: str
    shape: tuple[int, ...]
    dtype: str


class OnnxExportResult(BaseModel):
    onnx_path: Path
    sha256: str
    opset: int
    producer: str
    input_signature: OnnxInputSignature
    file_size_bytes: int
    validated: bool
    validation_notes: str | None = None


def _import_torch() -> Any:
    try:
        import torch  # noqa: F401
        return __import__("torch")
    except ImportError as e:
        raise RuntimeError(
            "torch is required for torch-module ONNX export but is not "
            "importable"
        ) from e


def _import_onnx() -> Any:
    try:
        return __import__("onnx")
    except ImportError as e:
        raise RuntimeError(
            "onnx is required for ONNX validation but is not importable"
        ) from e


def export_decoder_to_onnx(
    decoder: Any,
    *,
    output_path: Path,
    input_signature: OnnxInputSignature,
    opset: int = DEFAULT_OPSET,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    validate: bool = True,
) -> OnnxExportResult:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    producer = "decoderops.onnx_export"
    has_bytes = hasattr(decoder, "as_onnx_bytes") and callable(
        getattr(decoder, "as_onnx_bytes")
    )
    has_module = hasattr(decoder, "torch_module") and callable(
        getattr(decoder, "torch_module")
    )

    if has_bytes:
        blob = decoder.as_onnx_bytes()
        output_path.write_bytes(blob)
    elif has_module:
        torch = _import_torch()
        module = decoder.torch_module()
        dummy = torch.zeros(input_signature.shape, dtype=getattr(torch, input_signature.dtype))
        torch.onnx.export(
            module,
            dummy,
            str(output_path),
            opset_version=opset,
            do_constant_folding=True,
            export_params=True,
            input_names=[input_signature.name],
            dynamic_axes=dynamic_axes,
        )
    else:
        raise ValueError(
            "decoder must expose either .as_onnx_bytes() or .torch_module()"
        )

    validated = False
    notes: str | None = None
    if validate:
        try:
            onnx = _import_onnx()
            onnx.checker.check_model(str(output_path))
            validated = True
            notes = "onnx.checker.check_model passed"
        except RuntimeError:
            raise
        except Exception as e:  # noqa: BLE001
            validated = False
            notes = f"onnx.checker failed: {e}"

    sha = stamp_file(output_path)
    sidecar_sha = output_path.with_suffix(output_path.suffix + ".sha256")
    sidecar_sha.write_text(sha + "\n", encoding="utf-8")

    size = output_path.stat().st_size

    result = OnnxExportResult(
        onnx_path=output_path,
        sha256=sha,
        opset=opset,
        producer=producer,
        input_signature=input_signature,
        file_size_bytes=size,
        validated=validated,
        validation_notes=notes,
    )

    sidecar_manifest = output_path.with_suffix(
        output_path.suffix + ".manifest.json"
    )
    sidecar_manifest.write_text(
        json.dumps(
            result.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    return result
