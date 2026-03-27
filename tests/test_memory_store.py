import sqlite3
from pathlib import Path

from app.memory_store import MemoryStore


def test_memory_store_round_trip(tmp_path: Path):
    store = MemoryStore(tmp_path / "jarvis.db")

    store.store_memory(
        category="person",
        title="name",
        content="Asha",
        tags=["person"],
    )
    store.store_skill(
        name="Creation skill",
        description="Search and inspect files before responding.",
        trigger_hint="create",
        steps=[{"tool": "search_text"}],
    )
    store.record_interaction(
        "user",
        "I prefer quiet environments.",
        "remember",
        modality="audio",
        metadata={"transcript_source": "browser_speech"},
    )

    memories = store.recall("Asha")
    skills = store.list_recent_skills()
    interactions = store.list_recent_interactions()
    counts = store.get_memory_counts()
    markdown_files = list((tmp_path / "agent_memory").rglob("*.md"))

    assert memories[0]["content"] == "Asha"
    assert skills[0]["trigger_hint"] == "create"
    assert interactions[0]["modality"] == "audio"
    assert counts["memories"] == 1
    assert markdown_files


def test_memory_store_returns_runtime_context(tmp_path: Path):
    store = MemoryStore(tmp_path / "jarvis.db")

    store.store_memory(
        category="charter",
        title="duty",
        content="Adapt to operator priorities over time.",
        tags=["charter", "duty"],
    )
    store.store_memory(
        category="preference",
        title="preference",
        content="Keep replies concise.",
        tags=["preference"],
    )
    store.store_memory(
        category="experience",
        title="interaction",
        content="A transient interaction log.",
        tags=["experience"],
    )

    context = store.list_context_memories(limit=10)

    assert len(context) == 2
    assert {item["category"] for item in context} == {"charter", "preference"}


def test_legacy_db_memories_migrate_to_markdown(tmp_path: Path):
    db_path = tmp_path / "jarvis.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE memories(
                id INTEGER PRIMARY KEY,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                tags_json TEXT NOT NULL DEFAULT '[]',
                source TEXT NOT NULL DEFAULT 'system',
                confidence REAL NOT NULL DEFAULT 0.8,
                valid_from TEXT,
                valid_until TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            INSERT INTO memories(id, category, title, content, tags_json, source, confidence, created_at)
            VALUES(1, 'person', 'operator', 'Asha', '["person"]', 'legacy', 0.9, '2026-03-27 10:00:00')
            """
        )

    store = MemoryStore(db_path)

    memories = store.recall("Asha")
    markdown_files = list((tmp_path / "agent_memory" / "memories").rglob("*.md"))

    assert memories[0]["content"] == "Asha"
    assert markdown_files
