"""ReportsRepository — CRUD for Report (T067)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.report import Report

__all__ = ["ReportsRepository"]


class ReportsRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, report: Any) -> Report:
        obj = report if isinstance(report, Report) else Report(**dict(report))
        self.session.add(obj)
        self.session.flush()
        return obj

    def get_by_id(self, report_id: int) -> Report | None:
        return self.session.get(Report, report_id)

    def get_by_run_id(self, run_id: str) -> list[Report]:
        stmt = select(Report).where(Report.run_id == run_id)
        return list(self.session.execute(stmt).scalars())

    def get_latest_for_run(self, run_id: str, report_type: str) -> Report | None:
        stmt = (
            select(Report)
            .where(Report.run_id == run_id, Report.type == report_type)
            .order_by(Report.created_at.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def list_by_type(self, run_id: str, report_type: str) -> list[Report]:
        stmt = select(Report).where(
            Report.run_id == run_id, Report.type == report_type
        )
        return list(self.session.execute(stmt).scalars())

    def list_by_format(self, run_id: str, format: str) -> list[Report]:
        stmt = select(Report).where(
            Report.run_id == run_id, Report.format == format
        )
        return list(self.session.execute(stmt).scalars())
