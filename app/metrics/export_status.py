"""Per-format export status record (T045)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator

__all__ = ["ExportStatus", "build_export_status"]


class ExportStatus(BaseModel):
    format: Literal["onnx", "tensorrt"]
    attempted: bool
    succeeded: bool
    artefact_path: str | None = None
    error_message: str | None = None
    duration_seconds: float | None = None
    tool_version: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "ExportStatus":
        if self.succeeded:
            if self.artefact_path is None:
                raise ValueError(
                    "succeeded=True requires artefact_path to be set"
                )
            if self.error_message is not None:
                raise ValueError(
                    "succeeded=True forbids error_message being set"
                )
        else:
            if self.error_message is None:
                raise ValueError(
                    "succeeded=False requires error_message to be set"
                )
        return self


def build_export_status(
    format: str,
    attempted: bool,
    succeeded: bool,
    artefact_path: str | None = None,
    error_message: str | None = None,
    duration_seconds: float | None = None,
    tool_version: str | None = None,
) -> ExportStatus:
    return ExportStatus(
        format=format,  # type: ignore[arg-type]
        attempted=attempted,
        succeeded=succeeded,
        artefact_path=artefact_path,
        error_message=error_message,
        duration_seconds=duration_seconds,
        tool_version=tool_version,
    )
