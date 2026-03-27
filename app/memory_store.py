from __future__ import annotations

import json
import re
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MemoryStore:
    _CONTEXT_CATEGORIES = (
        "charter",
        "constraint",
        "interaction_style",
        "relationship",
        "objective",
        "person",
        "preference",
        "state",
        "context",
        "note",
    )
    _PERSONA_CATEGORIES = {
        "charter",
        "constraint",
        "interaction_style",
        "objective",
        "person",
        "preference",
        "relationship",
    }
    _WORKING_MEMORY_CATEGORIES = {"context", "experience", "note", "state"}

    def __init__(self, db_path: Path, brain_root: Path | None = None):
        self._db_path = db_path
        self._brain_root = (brain_root or (db_path.parent / "agent_brain")).resolve()
        self._workspace_root = self._brain_root / "workspace"
        self._knowledge_root = self._brain_root / "knowledge"
        self._memory_root = self._knowledge_root / "memories"
        self._persona_root = self._memory_root / "persona"
        self._working_root = self._memory_root / "working"
        self._long_term_root = self._memory_root / "long_term"
        self._skill_root = self._knowledge_root / "skills"
        self._insight_root = self._knowledge_root / "insights"
        self._brain_readme_path = self._brain_root / "README.md"
        self._workspace_readme_path = self._workspace_root / "README.md"
        self._persona_doc_path = self._brain_root / "persona.md"
        self._working_memory_doc_path = self._brain_root / "working-memory.md"
        self._long_term_doc_path = self._brain_root / "long-term-memory.md"
        self._skills_doc_path = self._brain_root / "skills.md"
        self._insights_doc_path = self._brain_root / "insights.md"
        self._workspace_map_doc_path = self._brain_root / "workspace-map.md"
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
        record_id = self._new_record_id()
        created_at = self._now_timestamp()
        metadata = {
            "id": record_id,
            "category": category,
            "title": title,
            "content": content,
            "tags": tags or [],
            "source": source,
            "confidence": confidence,
            "valid_from": valid_from,
            "valid_until": valid_until,
            "created_at": created_at,
            "storage_bucket": self._memory_bucket(category),
        }
        target = self._memory_path(record_id, category, title, created_at)
        body = f"# {title}\n\n{content}".strip() + "\n"
        self._write_markdown_record(target, metadata, body)
        self._sync_brain_documents()
        return record_id

    def recall(self, query: str = "", category: str | None = None, limit: int = 8) -> list[dict[str, Any]]:
        records = self._load_memory_records()
        if category:
            records = [record for record in records if record["category"] == category]

        tokens = self._search_tokens(query)
        if tokens:
            scored = []
            for record in records:
                score = self._memory_match_score(record, tokens)
                if score > 0:
                    scored.append((score, record))
            scored.sort(key=lambda item: (item[0], item[1]["created_at"], item[1]["id"]), reverse=True)
            return [record for _, record in scored[:limit]]

        records.sort(key=lambda record: (record["created_at"], record["id"]), reverse=True)
        return records[:limit]

    def list_recent_skills(self, limit: int = 6) -> list[dict[str, Any]]:
        records = self._load_skill_records()
        records.sort(
            key=lambda record: (
                record["last_used_at"] or record["created_at"],
                record["created_at"],
                record["id"],
            ),
            reverse=True,
        )
        return records[:limit]

    def list_context_memories(self, limit: int = 8) -> list[dict[str, Any]]:
        records = [
            record for record in self._load_memory_records() if record["category"] in self._CONTEXT_CATEGORIES
        ]
        records.sort(key=lambda record: (record["created_at"], record["id"]), reverse=True)
        return records[:limit]

    def store_skill(
        self,
        *,
        name: str,
        description: str,
        trigger_hint: str,
        steps: list[dict[str, Any]],
    ) -> int:
        target = self._skill_root / f"{self._slugify(trigger_hint)}--{self._slugify(name)}.md"
        existing = self._read_markdown_metadata(target) if target.exists() else {}
        timestamp = self._now_timestamp()
        metadata = {
            "id": int(existing.get("id") or self._new_record_id()),
            "name": name,
            "description": description,
            "trigger_hint": trigger_hint,
            "steps": steps,
            "success_count": int(existing.get("success_count", 0)) + 1,
            "created_at": existing.get("created_at") or timestamp,
            "last_used_at": timestamp if existing else None,
        }
        step_lines = "\n".join(f"- `{json.dumps(step, sort_keys=True)}`" for step in steps) or "- none"
        body = f"# {name}\n\n{description}\n\n## Steps\n{step_lines}\n"
        self._write_markdown_record(target, metadata, body)
        self._sync_brain_documents()
        return int(metadata["id"])

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
        record_id = self._new_record_id()
        created_at = self._now_timestamp()
        metadata = {
            "id": record_id,
            "severity": severity,
            "source": source,
            "title": title,
            "details": details,
            "file_path": file_path,
            "line_number": line_number,
            "status": status,
            "created_at": created_at,
        }
        target = self._insight_path(record_id, title, created_at)
        body = f"# {title}\n\n{details}".strip() + "\n"
        self._write_markdown_record(target, metadata, body)
        self._sync_brain_documents()
        return record_id

    def list_recent_insights(self, limit: int = 8) -> list[dict[str, Any]]:
        records = self._load_insight_records()
        records.sort(key=lambda record: (record["created_at"], record["id"]), reverse=True)
        return records[:limit]

    def get_memory_counts(self) -> dict[str, int]:
        with self._connect() as connection:
            counts = {
                "observations": int(connection.execute("SELECT COUNT(*) AS value FROM observations").fetchone()["value"]),
                "interactions": int(connection.execute("SELECT COUNT(*) AS value FROM interactions").fetchone()["value"]),
            }

        counts["memories"] = len(self._load_memory_records())
        counts["skills"] = len(self._load_skill_records())
        counts["insights"] = len(self._load_insight_records())
        return counts

    def _initialise(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy_brain_root()
        for directory in (
            self._brain_root,
            self._workspace_root,
            self._knowledge_root,
            self._memory_root,
            self._persona_root,
            self._working_root,
            self._long_term_root,
            self._skill_root,
            self._insight_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)

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

                CREATE TABLE IF NOT EXISTS observations(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    admin_present INTEGER NOT NULL,
                    face_count INTEGER NOT NULL DEFAULT 0,
                    brightness REAL NOT NULL DEFAULT 0,
                    note TEXT,
                    image_path TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            self._ensure_column(connection, "interactions", "modality", "TEXT NOT NULL DEFAULT 'text'")
            self._ensure_column(connection, "interactions", "channel", "TEXT NOT NULL DEFAULT 'ui'")
            self._ensure_column(connection, "interactions", "metadata_json", "TEXT NOT NULL DEFAULT '{}'")
            self._ensure_column(connection, "interactions", "media_path", "TEXT")
            self._migrate_legacy_knowledge_to_markdown(connection)
        self._sync_brain_documents()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_column(self, connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _migrate_legacy_knowledge_to_markdown(self, connection: sqlite3.Connection) -> None:
        if self._table_exists(connection, "memories"):
            rows = connection.execute(
                """
                SELECT id, category, title, content, tags_json, source, confidence, valid_from, valid_until, created_at
                FROM memories
                ORDER BY created_at ASC, id ASC
                """
            ).fetchall()
            for row in rows:
                created_at = self._normalize_timestamp(row["created_at"])
                target = self._memory_path(int(row["id"]), row["category"], row["title"], created_at)
                if target.exists():
                    continue
                metadata = {
                    "id": int(row["id"]),
                    "category": row["category"],
                    "title": row["title"],
                    "content": row["content"],
                    "tags": json.loads(row["tags_json"] or "[]"),
                    "source": row["source"],
                    "confidence": row["confidence"],
                    "valid_from": row["valid_from"],
                    "valid_until": row["valid_until"],
                    "created_at": created_at,
                    "storage_bucket": self._memory_bucket(row["category"]),
                }
                body = f"# {row['title']}\n\n{row['content']}".strip() + "\n"
                self._write_markdown_record(target, metadata, body)

        if self._table_exists(connection, "procedures"):
            rows = connection.execute(
                """
                SELECT id, name, description, trigger_hint, steps_json, success_count, created_at, last_used_at
                FROM procedures
                ORDER BY COALESCE(last_used_at, created_at) ASC, id ASC
                """
            ).fetchall()
            for row in rows:
                target = self._skill_root / f"{self._slugify(row['trigger_hint'])}--{self._slugify(row['name'])}.md"
                if target.exists():
                    continue
                metadata = {
                    "id": int(row["id"]),
                    "name": row["name"],
                    "description": row["description"],
                    "trigger_hint": row["trigger_hint"],
                    "steps": json.loads(row["steps_json"] or "[]"),
                    "success_count": int(row["success_count"]),
                    "created_at": self._normalize_timestamp(row["created_at"]),
                    "last_used_at": self._normalize_timestamp(row["last_used_at"]),
                }
                step_lines = "\n".join(
                    f"- `{json.dumps(step, sort_keys=True)}`" for step in metadata["steps"]
                ) or "- none"
                body = f"# {row['name']}\n\n{row['description']}\n\n## Steps\n{step_lines}\n"
                self._write_markdown_record(target, metadata, body)

        if self._table_exists(connection, "code_findings"):
            rows = connection.execute(
                """
                SELECT id, severity, source, title, details, file_path, line_number, status, created_at
                FROM code_findings
                ORDER BY created_at ASC, id ASC
                """
            ).fetchall()
            for row in rows:
                created_at = self._normalize_timestamp(row["created_at"])
                target = self._insight_path(int(row["id"]), row["title"], created_at)
                if target.exists():
                    continue
                metadata = {
                    "id": int(row["id"]),
                    "severity": row["severity"],
                    "source": row["source"],
                    "title": row["title"],
                    "details": row["details"],
                    "file_path": row["file_path"],
                    "line_number": row["line_number"],
                    "status": row["status"],
                    "created_at": created_at,
                }
                body = f"# {row['title']}\n\n{row['details']}".strip() + "\n"
                self._write_markdown_record(target, metadata, body)

    def _migrate_legacy_brain_root(self) -> None:
        legacy_root = self._db_path.parent / "agent_memory"
        if not legacy_root.exists() or legacy_root.resolve() == self._brain_root:
            return

        self._knowledge_root.mkdir(parents=True, exist_ok=True)
        mappings = (
            (legacy_root / "memories", self._memory_root),
            (legacy_root / "skills", self._skill_root),
            (legacy_root / "insights", self._insight_root),
        )
        for source, destination in mappings:
            if source.exists() and not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))

        if legacy_root.exists() and not any(legacy_root.iterdir()):
            legacy_root.rmdir()

    def _table_exists(self, connection: sqlite3.Connection, table: str) -> bool:
        row = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        return row is not None

    def _load_memory_records(self) -> list[dict[str, Any]]:
        records = []
        for path in self._memory_root.rglob("*.md"):
            metadata = self._read_markdown_metadata(path)
            if not metadata:
                continue
            records.append(self._coerce_memory_record(metadata))
        return records

    def _load_skill_records(self) -> list[dict[str, Any]]:
        records = []
        for path in self._skill_root.glob("*.md"):
            metadata = self._read_markdown_metadata(path)
            if not metadata:
                continue
            records.append(
                {
                    "id": int(metadata["id"]),
                    "name": metadata["name"],
                    "description": metadata["description"],
                    "trigger_hint": metadata["trigger_hint"],
                    "steps": list(metadata.get("steps", [])),
                    "success_count": int(metadata.get("success_count", 1)),
                    "created_at": self._normalize_timestamp(metadata.get("created_at")),
                    "last_used_at": self._normalize_timestamp(metadata.get("last_used_at")),
                }
            )
        return records

    def _load_insight_records(self) -> list[dict[str, Any]]:
        records = []
        for path in self._insight_root.glob("*.md"):
            metadata = self._read_markdown_metadata(path)
            if not metadata:
                continue
            records.append(
                {
                    "id": int(metadata["id"]),
                    "severity": metadata["severity"],
                    "source": metadata["source"],
                    "title": metadata["title"],
                    "details": metadata["details"],
                    "file_path": metadata.get("file_path"),
                    "line_number": metadata.get("line_number"),
                    "status": metadata.get("status", "open"),
                    "created_at": self._normalize_timestamp(metadata.get("created_at")),
                }
            )
        return records

    def _coerce_memory_record(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": int(metadata["id"]),
            "category": metadata["category"],
            "title": metadata["title"],
            "content": metadata["content"],
            "tags": list(metadata.get("tags", [])),
            "source": metadata.get("source", "system"),
            "confidence": float(metadata.get("confidence", 0.8)),
            "valid_from": metadata.get("valid_from"),
            "valid_until": metadata.get("valid_until"),
            "created_at": self._normalize_timestamp(metadata.get("created_at")),
        }

    def _write_markdown_record(self, path: Path, metadata: dict[str, Any], body: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["---"]
        for key, value in metadata.items():
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=True)}")
        lines.extend(["---", "", body.rstrip(), ""])
        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_markdown_document(self, path: Path, title: str, sections: list[str]) -> None:
        body = [f"# {title}", ""]
        for section in sections:
            if not section:
                continue
            body.append(section.rstrip())
            body.append("")
        path.write_text("\n".join(body).rstrip() + "\n", encoding="utf-8")

    def _read_markdown_metadata(self, path: Path) -> dict[str, Any]:
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            return {}

        remainder = text[4:]
        frontmatter, separator, _body = remainder.partition("\n---\n")
        if not separator:
            return {}

        metadata: dict[str, Any] = {}
        for line in frontmatter.splitlines():
            if not line.strip():
                continue
            key, separator, raw = line.partition(": ")
            if not separator:
                continue
            metadata[key] = json.loads(raw) if raw else None
        return metadata

    def _memory_match_score(self, record: dict[str, Any], tokens: list[str]) -> int:
        haystack = {
            "title": record["title"].lower(),
            "content": record["content"].lower(),
            "category": record["category"].lower(),
            "tags": " ".join(tag.lower() for tag in record.get("tags", [])),
        }
        score = 0
        for token in tokens:
            if token in haystack["title"]:
                score += 4
            if token in haystack["tags"]:
                score += 3
            if token in haystack["category"]:
                score += 2
            if token in haystack["content"]:
                score += 1
        return score

    def _memory_bucket(self, category: str) -> str:
        if category in self._PERSONA_CATEGORIES:
            return "persona"
        if category in self._WORKING_MEMORY_CATEGORIES:
            return "working"
        return "long_term"

    def _memory_directory(self, category: str) -> Path:
        bucket = self._memory_bucket(category)
        if bucket == "persona":
            return self._persona_root
        if bucket == "working":
            return self._working_root
        return self._long_term_root

    def _memory_path(self, record_id: int, category: str, title: str, created_at: str) -> Path:
        slug = self._slugify(title or category)
        timestamp = self._timestamp_slug(created_at)
        return self._memory_directory(category) / f"{timestamp}-{record_id}-{slug}.md"

    def _insight_path(self, record_id: int, title: str, created_at: str) -> Path:
        slug = self._slugify(title)
        timestamp = self._timestamp_slug(created_at)
        return self._insight_root / f"{timestamp}-{record_id}-{slug}.md"

    def _now_timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _normalize_timestamp(self, value: Any) -> str | None:
        if value in {None, ""}:
            return None
        if isinstance(value, datetime):
            dt = value
        else:
            text = str(value).strip()
            try:
                dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat(timespec="seconds")

    def _new_record_id(self) -> int:
        return time.time_ns() // 1_000

    def _timestamp_slug(self, created_at: str | None) -> str:
        normalized = self._normalize_timestamp(created_at) or self._now_timestamp()
        return re.sub(r"[^0-9A-Za-z]+", "", normalized)

    def _slugify(self, value: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower()).strip("-")
        return slug or "record"

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

    def _search_tokens(self, query: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9_]{2,}", query.lower())[:8]

    def _sync_brain_documents(self) -> None:
        memories = self._load_memory_records()
        skills = self._load_skill_records()
        insights = self._load_insight_records()

        persona_memories = [record for record in memories if self._memory_bucket(record["category"]) == "persona"]
        working_memories = [record for record in memories if self._memory_bucket(record["category"]) == "working"]
        long_term_memories = [record for record in memories if self._memory_bucket(record["category"]) == "long_term"]

        self._write_markdown_document(
            self._brain_readme_path,
            "Agent Brain",
            [
                "This directory is the agent's external brain. The agent may create, rearrange, and maintain files here.",
                "## Canonical Memory Files\n"
                "- `persona.md`: durable duties, preferences, relationship context, and behavioral identity.\n"
                "- `working-memory.md`: active notes, short-horizon state, and in-flight context.\n"
                "- `long-term-memory.md`: accumulated durable knowledge that does not fit persona.\n"
                "- `skills.md`: learned tool and execution patterns.\n"
                "- `insights.md`: evolution findings and codebase observations.\n"
                "- `workspace-map.md`: current tree of the freeform workspace.\n"
                "- `workspace/`: freeform directory tree the agent can organize as needed.",
            ],
        )
        if not self._workspace_readme_path.exists():
            self._write_markdown_document(
                self._workspace_readme_path,
                "Brain Workspace",
                [
                    "Use this directory as external working memory. Create folders, move notes, and reorganize structure as the brain evolves.",
                ],
            )

        self._write_markdown_document(
            self._persona_doc_path,
            "Persona",
            [
                "Durable identity, duties, operator preferences, and behavioral constraints.",
                self._render_memory_entries(persona_memories),
            ],
        )
        self._write_markdown_document(
            self._working_memory_doc_path,
            "Working Memory",
            [
                "Active state, recent notes, and short-horizon context.",
                self._render_memory_entries(working_memories),
            ],
        )
        self._write_markdown_document(
            self._long_term_doc_path,
            "Long-Term Memory",
            [
                "Durable knowledge accumulated over time that is not part of the agent persona.",
                self._render_memory_entries(long_term_memories),
            ],
        )
        self._write_markdown_document(
            self._skills_doc_path,
            "Learned Skills",
            [
                "Execution patterns captured from successful tool usage.",
                self._render_skill_entries(skills),
            ],
        )
        self._write_markdown_document(
            self._insights_doc_path,
            "Evolution Insights",
            [
                "Tracked self-improvement findings and codebase issues.",
                self._render_insight_entries(insights),
            ],
        )
        self._write_markdown_document(
            self._workspace_map_doc_path,
            "Workspace Map",
            [
                "Current tree of the freeform brain workspace.",
                self._render_workspace_tree(),
            ],
        )

    def _render_memory_entries(self, records: list[dict[str, Any]]) -> str:
        if not records:
            return "## Entries\n- none yet"
        lines = ["## Entries"]
        for record in sorted(records, key=lambda item: (item["created_at"], item["id"]), reverse=True)[:80]:
            tags = ", ".join(record.get("tags", []))
            detail = f"- [{record['category']}] **{record['title']}**: {record['content']}"
            if tags:
                detail += f" (`{tags}`)"
            detail += f" [{record['created_at']}]"
            lines.append(detail)
        return "\n".join(lines)

    def _render_skill_entries(self, records: list[dict[str, Any]]) -> str:
        if not records:
            return "## Entries\n- none yet"
        lines = ["## Entries"]
        for record in records[:80]:
            lines.append(
                f"- **{record['name']}** (`{record['trigger_hint']}`): {record['description']} "
                f"[successes: {record['success_count']}]"
            )
        return "\n".join(lines)

    def _render_insight_entries(self, records: list[dict[str, Any]]) -> str:
        if not records:
            return "## Entries\n- none yet"
        lines = ["## Entries"]
        for record in records[:80]:
            lines.append(
                f"- **{record['severity']}** / **{record['status']}** / **{record['title']}**: {record['details']}"
            )
        return "\n".join(lines)

    def _render_workspace_tree(self) -> str:
        lines = ["## Tree", f"- {self._workspace_root.name}/"]
        entries = []
        for candidate in sorted(self._workspace_root.rglob("*")):
            relative = candidate.relative_to(self._workspace_root)
            if relative.parts == ("README.md",):
                continue
            depth = len(relative.parts)
            prefix = "  " * depth
            suffix = "/" if candidate.is_dir() else ""
            entries.append(f"{prefix}- {relative.name}{suffix}")
            if len(entries) >= 120:
                entries.append("- ...")
                break
        if not entries:
            entries.append("  - README.md")
        return "\n".join(lines + entries)
