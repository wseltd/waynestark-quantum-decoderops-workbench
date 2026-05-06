"""Markdown renderer for report templates (T083)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jinja2

__all__ = ["render_markdown"]

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _env() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=False,  # markdown never auto-HTML-escapes
        keep_trailing_newline=True,
        lstrip_blocks=True,
        trim_blocks=True,
    )


def render_markdown(template_name: str, context: dict[str, Any]) -> str:
    """Render `{template_name}.md.j2` with the given context."""
    env = _env()
    tmpl = env.get_template(f"{template_name}.md.j2")
    return tmpl.render(**context)
