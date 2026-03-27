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

    assert memories[0]["content"] == "Asha"
    assert skills[0]["trigger_hint"] == "create"
    assert interactions[0]["modality"] == "audio"
    assert counts["memories"] == 1


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
