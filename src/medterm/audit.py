from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from medterm.models import MatchResponse, ReviewRecord, ReviewRequest


class AuditStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS match_records (
                    request_id TEXT PRIMARY KEY,
                    artifact_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS review_records (
                    review_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    span_id TEXT NOT NULL,
                    reviewer_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    concept_id TEXT,
                    note TEXT,
                    artifact_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(request_id) REFERENCES match_records(request_id)
                );
                CREATE INDEX IF NOT EXISTS idx_review_request_span
                ON review_records(request_id, span_id);
                """
            )

    def save_match(self, response: MatchResponse) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO match_records
                (request_id, artifact_version, created_at, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    response.request_id,
                    response.artifact_version,
                    response.created_at.isoformat(),
                    response.model_dump_json(),
                ),
            )

    def get_match(self, request_id: str) -> MatchResponse | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM match_records WHERE request_id = ?", (request_id,)
            ).fetchone()
        return MatchResponse.model_validate_json(row["payload_json"]) if row else None

    def save_review(
        self, request_id: str, span_id: str, review: ReviewRequest, artifact_version: str
    ) -> ReviewRecord:
        created_at = datetime.now(UTC)
        record = ReviewRecord(
            review_id=str(uuid.uuid4()),
            request_id=request_id,
            span_id=span_id,
            reviewer_id=review.reviewer_id,
            action=review.action,
            concept_id=review.concept_id,
            note=review.note,
            artifact_version=artifact_version,
            created_at=created_at,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO review_records
                (review_id, request_id, span_id, reviewer_id, action, concept_id, note,
                 artifact_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.review_id,
                    request_id,
                    span_id,
                    review.reviewer_id,
                    review.action,
                    review.concept_id,
                    review.note,
                    artifact_version,
                    created_at.isoformat(),
                ),
            )
        return record

    def list_reviews(self, request_id: str) -> list[ReviewRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM review_records WHERE request_id = ? ORDER BY created_at",
                (request_id,),
            ).fetchall()
        return [ReviewRecord.model_validate(dict(row)) for row in rows]

    def export_jsonl(self, path: Path) -> int:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT m.payload_json, r.review_id, r.span_id, r.reviewer_id, r.action,
                       r.concept_id, r.note, r.created_at AS review_created_at
                FROM match_records m
                LEFT JOIN review_records r ON r.request_id = m.request_id
                ORDER BY m.created_at, r.created_at
                """
            ).fetchall()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                item = json.loads(row["payload_json"])
                item["review"] = {
                    key: row[key]
                    for key in (
                        "review_id",
                        "span_id",
                        "reviewer_id",
                        "action",
                        "concept_id",
                        "note",
                        "review_created_at",
                    )
                    if row[key] is not None
                }
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        return len(rows)
