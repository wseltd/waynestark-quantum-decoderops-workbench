"""Uvicorn-compatible ASGI entry point.

The Definition of Done's ``local_proof_cmd`` targets
``app.api.main:app``. This module instantiates the FastAPI app once so
``uvicorn app.api.main:app`` works without the factory flag.

For factory-style use (testing with injected Settings) prefer
``uvicorn app.api.app:create_app --factory``.
"""

from __future__ import annotations

from app.api.app import create_app

app = create_app()

__all__ = ["app"]
