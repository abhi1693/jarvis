from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any


class MemoryStore:
    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._initialise()

    def record_interaction(
        self,
        role: str,
        content: str,
        intent: str,
        *,
        modality: str = "text",
        channel: str = "ui",
        metadata: dict[str, Any] | None = None,
        media_path: str | None = None,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO interactions(role, content, intent, modality, channel, metadata_json, media_path)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (role, content, intent, modality, channel, json.dumps(metadata or {}), media_path),
            )
            connection.execute(
                """
                INSERT INTO interaction_fts(rowid, role, content, intent)
                VALUES(?, ?, ?, ?)
                """,
                (cursor.lastrowid, role, content, intent),
            )
            return int(cursor.lastrowid)

    def list_recent_interactions(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, role, content, intent, modality, channel, metadata_json, media_path, created_at
                FROM interactions
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_interaction(row) for row in rows]

    def store_memory(
        self,
        *,
        category: str,
        title: str,
        content: str,
        tags: list[str] | None = None,
        source: str = "system",
        confidence: float = 0.8,
        valid_from: str | None = None,
        valid_until: str | None = None,
    ) -> int:
        tag_json = json.dumps(tags or [])
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO memories(category, title, content, tags_json, source, confidence, valid_from, valid_until)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (category, title, content, tag_json, source, confidence, valid_from, valid_until),
            )
            connection.execute(
                """
                INSERT INTO memory_fts(rowid, title, content, tags)
                VALUES(?, ?, ?, ?)
                """,
                (cursor.lastrowid, title, content, " ".join(tags or [])),
            )
            return int(cursor.lastrowid)

    def recall(self, query: str = "", category: str | None = None, limit: int = 8) -> list[dict[str, Any]]:
        sql = """
            SELECT id, category, title, content, tags_json, source, confidence, valid_from, valid_until, created_at
            FROM memories
        """
        params: list[Any] = []
        clauses: list[str] = []
        safe_query = self._build_safe_fts_query(query)

        if safe_query:
            sql = """
                SELECT m.id, m.category, m.title, m.content, m.tags_json, m.source, m.confidence,
                       m.valid_from, m.valid_until, m.created_at
                FROM memory_fts f
                JOIN memories m ON m.id = f.rowid
            """
            clauses.append("memory_fts MATCH ?")
            params.append(safe_query)

        if category:
            clauses.append(("m." if safe_query else "") + "category = ?")
            params.append(category)

        if clauses:
            sql += " WHERE " + " AND ".join(clauses)

        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)

        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()

        return [self._row_to_memory(row) for row in rows]

    def list_recent_skills(self, limit: int = 6) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, name, description, trigger_hint, steps_json, success_count, created_at, last_used_at
                FROM procedures
                ORDER BY COALESCE(last_used_at, created_at) DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_skill(row) for row in rows]

    def store_skill(
        self,
        *,
        name: str,
        description: str,
        trigger_hint: str,
        steps: list[dict[str, Any]],
    ) -> int:
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id, success_count FROM procedures WHERE name = ? AND trigger_hint = ?",
                (name, trigger_hint),
            ).fetchone()
            if existing:
                connection.execute(
                    """
                    UPDATE procedures
                    SET description = ?, steps_json = ?, success_count = ?, last_used_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (description, json.dumps(steps), existing["success_count"] + 1, existing["id"]),
                )
                return int(existing["id"])

            cursor = connection.execute(
                """
                INSERT INTO procedures(name, description, trigger_hint, steps_json, success_count)
                VALUES(?, ?, ?, ?, 1)
                """,
                (name, description, trigger_hint, json.dumps(steps)),
            )
            return int(cursor.lastrowid)

    def store_observation(
        self,
        *,
        admin_present: bool,
        face_count: int,
        brightness: float,
        note: str | None = None,
        image_path: str | None = None,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO observations(admin_present, face_count, brightness, note, image_path)
                VALUES(?, ?, ?, ?, ?)
                """,
                (int(admin_present), face_count, brightness, note, image_path),
            )
            return int(cursor.lastrowid)

    def get_last_observation(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, admin_present, face_count, brightness, note, image_path, created_at
                FROM observations
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "admin_present": bool(row["admin_present"]),
            "face_count": row["face_count"],
            "brightness": row["brightness"],
            "note": row["note"],
            "image_path": row["image_path"],
            "created_at": row["created_at"],
        }

    def store_evolution_insight(
        self,
        *,
        severity: str,
        source: str,
        title: str,
        details: str,
        file_path: str | None = None,
        line_number: int | None = None,
        status: str = "open",
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO code_findings(severity, source, title, details, file_path, line_number, status)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (severity, source, title, details, file_path, line_number, status),
            )
            return int(cursor.lastrowid)

    def list_recent_insights(self, limit: int = 8) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, severity, source, title, details, file_path, line_number, status, created_at
                FROM code_findings
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_memory_counts(self) -> dict[str, int]:
        with self._connect() as connection:
            tables = {
                "memories": "SELECT COUNT(*) AS value FROM memories",
                "skills": "SELECT COUNT(*) AS value FROM procedures",
                "observations": "SELECT COUNT(*) AS value FROM observations",
                "insights": "SELECT COUNT(*) AS value FROM code_findings",
                "interactions": "SELECT COUNT(*) AS value FROM interactions",
            }
            return {
                name: int(connection.execute(query).fetchone()["value"])
                for name, query in tables.items()
            }

    def _initialise(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS interactions(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS interaction_fts USING fts5(
                    role,
                    content,
                    intent
                );

                CREATE TABLE IF NOT EXISTS memories(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    source TEXT NOT NULL DEFAULT 'system',
                    confidence REAL NOT NULL DEFAULT 0.8,
                    valid_from TEXT,
                    valid_until TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    title,
                    content,
                    tags
                );

                CREATE TABLE IF NOT EXISTS procedures(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    trigger_hint TEXT NOT NULL,
                    steps_json TEXT NOT NULL,
                    success_count INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_used_at TEXT
                );

                CREATE TABLE IF NOT EXISTS observations(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    admin_present INTEGER NOT NULL,
                    face_count INTEGER NOT NULL DEFAULT 0,
                    brightness REAL NOT NULL DEFAULT 0,
                    note TEXT,
                    image_path TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS code_findings(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    severity TEXT NOT NULL,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    details TEXT NOT NULL,
                    file_path TEXT,
                    line_number INTEGER,
                    status TEXT NOT NULL DEFAULT 'open',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            self._ensure_column(connection, "interactions", "modality", "TEXT NOT NULL DEFAULT 'text'")
            self._ensure_column(connection, "interactions", "channel", "TEXT NOT NULL DEFAULT 'ui'")
            self._ensure_column(connection, "interactions", "metadata_json", "TEXT NOT NULL DEFAULT '{}'")
            self._ensure_column(connection, "interactions", "media_path", "TEXT")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_column(self, connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _row_to_memory(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "category": row["category"],
            "title": row["title"],
            "content": row["content"],
            "tags": json.loads(row["tags_json"] or "[]"),
            "source": row["source"],
            "confidence": row["confidence"],
            "valid_from": row["valid_from"],
            "valid_until": row["valid_until"],
            "created_at": row["created_at"],
        }

    def _row_to_skill(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "trigger_hint": row["trigger_hint"],
            "steps": json.loads(row["steps_json"]),
            "success_count": row["success_count"],
            "created_at": row["created_at"],
            "last_used_at": row["last_used_at"],
        }

    def _row_to_interaction(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "intent": row["intent"],
            "modality": row["modality"],
            "channel": row["channel"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
            "media_path": row["media_path"],
            "created_at": row["created_at"],
        }

    def _build_safe_fts_query(self, query: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9_]{2,}", query.lower())
        if not tokens:
            return ""
        return " OR ".join(f'"{token}"' for token in tokens[:8])
