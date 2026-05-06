"""MetricsRepository — CRUD for Metrics (T065)."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.metrics import Metrics

__all__ = ["MetricNotFoundError", "MetricsRepository"]

_LOG = logging.getLogger(__name__)


class MetricNotFoundError(LookupError):
    pass


class MetricsRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, metric: Any) -> Metrics:
        obj = metric if isinstance(metric, Metrics) else Metrics(**dict(metric))
        self.session.add(obj)
        self.session.flush()
        _LOG.debug(
            "metric created: run_id=%s metrics_id=%s", obj.run_id, obj.metrics_id
        )
        return obj

    def get_by_id(self, metric_id: int) -> Metrics | None:
        return self.session.get(Metrics, metric_id)

    def get_by_run_id(self, run_id: str) -> list[Metrics]:
        stmt = select(Metrics).where(Metrics.run_id == run_id)
        return list(self.session.execute(stmt).scalars())

    def bulk_insert(self, metrics: list[Any]) -> list[Metrics]:
        objs = [
            m if isinstance(m, Metrics) else Metrics(**dict(m))
            for m in metrics
        ]
        self.session.add_all(objs)
        self.session.flush()
        _LOG.debug("bulk_insert metrics: n=%s", len(objs))
        return objs
