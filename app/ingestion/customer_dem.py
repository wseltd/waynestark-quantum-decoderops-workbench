"""Customer DEM bundle parser (T019) — thin wrapper over dem_parser."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ingestion.dem_parser import parse_dem_file

__all__ = ["parse_customer_dem_bundle"]


def parse_customer_dem_bundle(
    bundle_path: Path,
    customer_label: str,
    source_label: str | None = None,
    *,
    ingester_version: str = "0.1.0",
) -> dict[str, Any]:
    """Parse a customer-supplied DEM bundle (directory of .dem files)."""
    bundle_path = Path(bundle_path).expanduser().resolve()
    if not bundle_path.exists():
        raise FileNotFoundError(f"bundle missing: {bundle_path}")
    results = []
    if bundle_path.is_file():
        files = [bundle_path]
    else:
        files = sorted(bundle_path.glob("*.dem"))
    for dem in files:
        normalised = parse_dem_file(dem, ingester_version=ingester_version)
        results.append(
            {"path": str(dem), "normalised": normalised.model_dump(mode="json")}
        )
    return {
        "customer_label": customer_label,
        "source_label": source_label,
        "count": len(results),
        "entries": results,
    }
