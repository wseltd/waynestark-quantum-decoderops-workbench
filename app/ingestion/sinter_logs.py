"""Sinter shot-log parser (T018)."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

__all__ = ["InvalidSinterLogError", "parse_sinter_shot_log"]


class InvalidSinterLogError(ValueError):
    pass


def _stream_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_sinter_shot_log(
    path: Path, source_label: str | None = None
) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise InvalidSinterLogError(f"empty shot log at {path}")

    first_line = next(
        (ln for ln in text.splitlines() if ln.strip()), ""
    )
    fmt = "jsonl" if first_line.strip().startswith("{") else "csv"
    tasks: list[dict[str, Any]] = []
    line_count = 0
    if fmt == "jsonl":
        for i, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            line_count += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise InvalidSinterLogError(
                    f"malformed JSONL at line {i}: {e}"
                ) from e
            tasks.append(rec)
    else:
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames is None:
            raise InvalidSinterLogError(
                f"sinter CSV at {path} has no header row"
            )
        # sinter emits whitespace-padded column headers like
        # ' shots,    errors, ...'; normalise before the schema check.
        normalised_fieldnames = [
            (fn.strip() if isinstance(fn, str) else fn)
            for fn in reader.fieldnames
        ]
        required = {"shots", "errors"}
        missing_cols = required - set(normalised_fieldnames)
        if missing_cols:
            raise InvalidSinterLogError(
                f"sinter CSV at {path} missing required column(s) "
                f"{sorted(missing_cols)}; saw columns={normalised_fieldnames}"
            )
        for i, raw_row in enumerate(reader, 2):
            line_count += 1
            row = {
                (k.strip() if isinstance(k, str) else k): (
                    v.strip() if isinstance(v, str) else v
                )
                for k, v in raw_row.items()
            }
            out: dict[str, Any] = dict(row)
            if row.get("json_metadata"):
                try:
                    out["json_metadata"] = json.loads(row["json_metadata"])
                except json.JSONDecodeError as e:
                    raise InvalidSinterLogError(
                        f"bad json_metadata at line {i}: {e}"
                    ) from e
            tasks.append(out)

    shots_total = 0
    errors_total = 0
    discards_total = 0
    seconds_total = 0.0
    for idx, t in enumerate(tasks, 1):
        # `shots` and `errors` must be present on every row; the header
        # check above guards the CSV branch and every JSONL record is
        # expected to carry both. A missing value is a real corruption
        # signal — do not silently degrade to zero.
        if "shots" not in t or t["shots"] in (None, ""):
            raise InvalidSinterLogError(
                f"row {idx}: missing 'shots' value"
            )
        if "errors" not in t or t["errors"] in (None, ""):
            raise InvalidSinterLogError(
                f"row {idx}: missing 'errors' value"
            )
        try:
            s = int(t["shots"])
            e = int(t["errors"])
            d = int(t.get("discards", 0) or 0)
            sec = float(t.get("seconds", 0) or 0)
        except (TypeError, ValueError) as err:
            raise InvalidSinterLogError(
                f"row {idx}: non-integer shots/errors/discards: {err}"
            ) from err
        if s < 0 or e < 0 or d < 0:
            raise InvalidSinterLogError(
                f"row {idx}: negative shots/errors/discards: {t}"
            )
        if e > s:
            raise InvalidSinterLogError(
                f"row {idx}: errors ({e}) exceed shots ({s})"
            )
        shots_total += s
        errors_total += e
        discards_total += d
        seconds_total += sec
    return {
        "input_source": "sinter_shot_log",
        "schema_version": "1",
        "tasks": tasks,
        "shots_total": shots_total,
        "errors_total": errors_total,
        "discards_total": discards_total,
        "seconds_total": seconds_total,
        "provenance": {
            "absolute_path": str(path),
            "sha256": _stream_sha256(path),
            "file_size": path.stat().st_size,
            "line_count": line_count,
            "format_detected": fmt,
        },
    }
