from pathlib import Path

from app.memory_store import MemoryStore


def test_memory_store_round_trip(tmp_path: Path):
    store = MemoryStore(tmp_path / "jarvis.db")

    store.store_memory(
        category="profile",
        title="name",
        content="Asha",
        tags=["person", "profile"],
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

    assert memories[0]["content"] == "Asha"
    assert skills[0]["trigger_hint"] == "create"
    assert interactions[0]["modality"] == "audio"
    assert counts["memories"] == 1
