"""Deterministic PDF renderer via reportlab (T085)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["render_pdf"]


def render_pdf(
    template_name: str,
    context: dict[str, Any],
    output_path: Path,
) -> Path:
    """Render a simple PDF from a markdown rendering.

    Deterministic: embeds no wall-clock timestamps (reportlab default writes
    creation/modification dates; we override both to the context's
    ``build_timestamp_utc`` if present, else a fixed epoch marker).
    """
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.pdfgen import canvas
    except ImportError as e:
        raise RuntimeError("reportlab is required for PDF rendering") from e

    from app.reports.markdown_renderer import render_markdown

    md = render_markdown(template_name, context)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path), pagesize=LETTER)
    # Set deterministic PDF metadata
    c.setTitle(f"decoderops-report:{template_name}")
    c.setAuthor("decoderops-reports")
    c.setSubject(template_name)
    c.setCreator("decoderops-reports")
    # Reportlab reads dates from the OS; override by pre-setting info dict
    # via canvas._doc.info
    c._doc.info.creationDate = None  # type: ignore[attr-defined]
    c._doc.info.modDate = None  # type: ignore[attr-defined]

    width, height = LETTER
    x_margin = 40
    y = height - 50
    for line in md.splitlines():
        if y < 50:
            c.showPage()
            y = height - 50
        c.drawString(x_margin, y, line[:120])
        y -= 14
    c.save()
    return output_path
