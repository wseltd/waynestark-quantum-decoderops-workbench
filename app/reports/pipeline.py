"""Report pipeline — render the 6×4 matrix of reports (T088)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.reports.html_renderer import render_html
from app.reports.json_renderer import render_json
from app.reports.markdown_renderer import render_markdown

__all__ = ["REPORT_TYPES", "RenderedReport", "render_all"]


REPORT_TYPES: tuple[str, ...] = (
    "engineering_benchmark",
    "decoder_comparison",
    "deployment_readiness",
    "artefact_manifest",
    "risk_caveat",
)

# Decision reports are rendered via a separate pipeline entry so callers
# can opt in without needing to stage a full 5×4 context. See
# render_decision_report below.
DECISION_REPORT_TYPE = "decision_report"


@dataclass(frozen=True)
class RenderedReport:
    type: str
    format: str
    path: Path
    sha256: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render_all(
    *,
    context: dict[str, Any],
    output_dir: Path,
    include_pdf: bool = False,
) -> list[RenderedReport]:
    """Render every (type, format) pair deterministically to ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[RenderedReport] = []

    for rtype in REPORT_TYPES:
        md = render_markdown(rtype, context)
        md_path = output_dir / f"{rtype}.md"
        md_bytes = md.encode("utf-8")
        md_path.write_bytes(md_bytes)
        results.append(RenderedReport(rtype, "markdown", md_path, _sha256(md_bytes)))

        html = render_html(rtype, context)
        html_path = output_dir / f"{rtype}.html"
        html_bytes = html.encode("utf-8")
        html_path.write_bytes(html_bytes)
        results.append(RenderedReport(rtype, "html", html_path, _sha256(html_bytes)))

        j = render_json({"type": rtype, "context": context})
        j_path = output_dir / f"{rtype}.json"
        j_bytes = (j + "\n").encode("utf-8")
        j_path.write_bytes(j_bytes)
        results.append(RenderedReport(rtype, "json", j_path, _sha256(j_bytes)))

        if include_pdf:
            from app.reports.pdf_renderer import render_pdf

            pdf_path = output_dir / f"{rtype}.pdf"
            render_pdf(rtype, context, pdf_path)
            pdf_bytes = pdf_path.read_bytes()
            results.append(
                RenderedReport(rtype, "pdf", pdf_path, _sha256(pdf_bytes))
            )

    return results


def render_decision_report(
    *,
    decision_context: dict[str, Any],
    output_dir: Path,
    include_pdf: bool = False,
) -> list[RenderedReport]:
    """Render the decision report in md/html/json (+ optional pdf).

    ``decision_context`` must include keys ``profile``, ``decision``,
    ``run``, ``provenance`` as produced by
    :meth:`app.profiles.runner.ProfileRunResult.to_dict` + a wrapping
    step in the caller. The JSON output echoes the full structured
    decision so downstream tooling (API, CLI, evidence pack) can parse
    it without re-rendering.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[RenderedReport] = []

    md = render_markdown(DECISION_REPORT_TYPE, decision_context)
    md_path = output_dir / f"{DECISION_REPORT_TYPE}.md"
    md_bytes = md.encode("utf-8")
    md_path.write_bytes(md_bytes)
    results.append(
        RenderedReport(DECISION_REPORT_TYPE, "markdown", md_path, _sha256(md_bytes))
    )

    html = render_html(DECISION_REPORT_TYPE, decision_context)
    html_path = output_dir / f"{DECISION_REPORT_TYPE}.html"
    html_bytes = html.encode("utf-8")
    html_path.write_bytes(html_bytes)
    results.append(
        RenderedReport(DECISION_REPORT_TYPE, "html", html_path, _sha256(html_bytes))
    )

    j = render_json({"type": DECISION_REPORT_TYPE, "context": decision_context})
    j_path = output_dir / f"{DECISION_REPORT_TYPE}.json"
    j_bytes = (j + "\n").encode("utf-8")
    j_path.write_bytes(j_bytes)
    results.append(
        RenderedReport(DECISION_REPORT_TYPE, "json", j_path, _sha256(j_bytes))
    )

    if include_pdf:
        from app.reports.pdf_renderer import render_pdf

        pdf_path = output_dir / f"{DECISION_REPORT_TYPE}.pdf"
        render_pdf(DECISION_REPORT_TYPE, decision_context, pdf_path)
        pdf_bytes = pdf_path.read_bytes()
        results.append(
            RenderedReport(DECISION_REPORT_TYPE, "pdf", pdf_path, _sha256(pdf_bytes))
        )

    return results
