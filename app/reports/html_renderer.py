"""HTML renderer for report templates (T084)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jinja2

__all__ = ["render_html"]

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def render_html(template_name: str, context: dict[str, Any]) -> str:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=jinja2.select_autoescape(["html", "htm", "j2", "html.j2"]),
        keep_trailing_newline=True,
    )
    tmpl = env.get_template(f"{template_name}.html.j2")
    return tmpl.render(**context)
