"""HTTP client for the CLI — wraps httpx.Client (T105)."""

from __future__ import annotations

import os
from typing import Any

import httpx

__all__ = ["DecoderOpsClient", "DecoderOpsClientError"]


DEFAULT_BASE_URL = "http://127.0.0.1:8000"


class DecoderOpsClientError(RuntimeError):
    pass


class DecoderOpsClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        if base_url is None:
            base_url = os.environ.get(
                "DECODEROPS_API_BASE_URL", DEFAULT_BASE_URL
            )
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def request(
        self,
        method: str,
        path: str,
        json: Any | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            r = self._client.request(
                method, path, json=json, params=params
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise DecoderOpsClientError(
                f"HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise DecoderOpsClientError(
                f"request failed: {e}"
            ) from e
        try:
            return r.json()
        except ValueError as e:
            raise DecoderOpsClientError(
                f"invalid JSON response: {e}"
            ) from e

    def get_health(self) -> dict[str, Any]:
        return self.request("GET", "/health")

    def post_seed(self, seed: int, num_workers: int = 1) -> dict[str, Any]:
        return self.request(
            "POST",
            "/seed",
            params={"master_seed": seed, "num_workers": num_workers},
        )

    def close(self) -> None:
        self._client.close()
