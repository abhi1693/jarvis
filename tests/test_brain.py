import asyncio
from pathlib import Path

from app.brain import BrainService
from app.config import Settings
from app.memory_store import MemoryStore
from app.tools.filesystem import FilesystemTool


class StubLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    @property
    def enabled(self) -> bool:
        return True

    async def complete_json(self, _system_prompt: str, _user_prompt: str):
        if not self._responses:
            return None
        return self._responses.pop(0)


def make_settings(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    brain_root = data_dir / "agent_brain"
    brain_workspace_dir = brain_root / "workspace"
    brain_skill_dir = brain_workspace_dir / "skills"
    settings = Settings(
        app_name="Adaptive Agent",
        repo_root=tmp_path,
        data_dir=data_dir,
        db_path=data_dir / "jarvis.db",
        brain_root=brain_root,
        brain_workspace_dir=brain_workspace_dir,
        brain_skill_dir=brain_skill_dir,
        brain_skill_source_dirs=(),
        admin_face_path=data_dir / "admin_face.npy",
        snapshot_dir=data_dir / "snapshots",
        media_dir=data_dir / "media",
        change_set_dir=data_dir / "change_sets",
        command_timeout_seconds=20,
        memory_recall_limit=8,
        brain_refresh_interval_seconds=1,
        brain_working_memory_ttl_seconds=60,
        llm_compat_url="http://llm.test",
        llm_model="test-model",
        llm_api_key=None,
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.brain_root.mkdir(parents=True, exist_ok=True)
    settings.brain_workspace_dir.mkdir(parents=True, exist_ok=True)
    settings.brain_skill_dir.mkdir(parents=True, exist_ok=True)
    settings.snapshot_dir.mkdir(parents=True, exist_ok=True)
    settings.media_dir.mkdir(parents=True, exist_ok=True)
    settings.change_set_dir.mkdir(parents=True, exist_ok=True)
    return settings


def test_brain_service_ingest_user_message_stores_selected_memory(tmp_path: Path):
    settings = make_settings(tmp_path)
    store = MemoryStore(settings.db_path, settings.brain_root)
    fs_tool = FilesystemTool(settings.repo_root, settings.brain_root)
    llm = StubLLM(
        [
            {
                "remember": [
                    {
                        "category": "preference",
                        "title": "reply_style",
                        "content": "Keep replies quiet and terse.",
                        "tags": ["preference", "style"],
                        "confidence": 0.93,
                    }
                ],
                "workspace_actions": [
                    {
                        "action": "write_file",
                        "path": "attention.md",
                        "content": "# Attention\n- Favor terse replies.\n",
                    }
                ],
                "forget_ids": [],
            }
        ]
    )
    brain = BrainService(settings, store, llm, fs_tool)

    result = asyncio.run(
        brain.ingest_user_message(
            message="Please keep replies quiet and terse.",
            modality="audio",
            intent_name="orient",
            memory_candidates=[],
        )
    )

    assert result["remembered"][0]["category"] == "preference"
    assert store.recall("terse", limit=5)[0]["content"] == "Keep replies quiet and terse."
    assert (settings.brain_workspace_dir / "attention.md").exists()


def test_brain_service_learns_skill_and_reads_workspace_files(tmp_path: Path):
    settings = make_settings(tmp_path)
    store = MemoryStore(settings.db_path, settings.brain_root)
    fs_tool = FilesystemTool(settings.repo_root, settings.brain_root)
    brain = BrainService(settings, store, StubLLM([]), fs_tool)

    store.store_memory(
        category="charter",
        title="duty",
        content="Adapt to operator priorities over time.",
        tags=["charter", "duty"],
    )
    fs_tool.write_file(
        "data/agent_brain/workspace/projects.md",
        "# Projects\n- Build a self-managing brain.\n",
    )
    brain.learn_from_tool_trace(
        "create",
        [{"tool": "search_text", "pattern": "brain service", "result": {"ok": True}}],
        "build the brain service",
    )

    prompt_context = brain.build_prompt_context(
        "self-managing brain",
        runtime_context=store.list_context_memories(limit=8),
        recalled_memories=store.recall("brain", limit=8),
        skills=store.list_recent_skills(limit=6),
    )

    assert "Adapt to operator priorities over time." in prompt_context
    assert "Build a self-managing brain." in prompt_context
    assert "Create skill" in prompt_context or "Create skill".lower() in prompt_context.lower()
    assert (settings.brain_workspace_dir / "skill-journal.md").exists()


def test_brain_service_reads_imported_and_local_skill_files(tmp_path: Path):
    settings = make_settings(tmp_path)
    external_root = tmp_path / ".codex" / "skills"
    external_skill = external_root / "python-helper" / "SKILL.md"
    external_skill.parent.mkdir(parents=True, exist_ok=True)
    (external_skill.parent / "references").mkdir(parents=True, exist_ok=True)
    (external_skill.parent / "scripts").mkdir(parents=True, exist_ok=True)
    external_skill.write_text(
        "# Python Helper\n\nUse this skill when working on Python services and tests.\n",
        encoding="utf-8",
    )
    (external_skill.parent / "references" / "pytest.md").write_text(
        "Prefer pytest fixtures for shared setup.\n",
        encoding="utf-8",
    )
    (external_skill.parent / "scripts" / "check.py").write_text(
        "print('python helper')\n",
        encoding="utf-8",
    )
    settings.brain_skill_source_dirs = (external_root,)

    store = MemoryStore(settings.db_path, settings.brain_root)
    store.sync_external_skill_library(settings.brain_skill_source_dirs)
    fs_tool = FilesystemTool(settings.repo_root, settings.brain_root)
    fs_tool.write_file(
        "data/agent_brain/workspace/skills/release.md",
        "# Release Skill\n\nUse this when preparing a release or changelog.\n",
    )
    brain = BrainService(settings, store, StubLLM([]), fs_tool)

    prompt_context = brain.build_prompt_context(
        "python release",
        runtime_context=[],
        recalled_memories=[],
        skills=store.list_recent_skills(limit=6),
    )

    assert "python-helper" in prompt_context
    assert "Release Skill" in prompt_context
    assert (settings.brain_root / "library" / "skills").exists()
    assert (
        settings.brain_root / "library" / "skills" / "codex-skills" / "python-helper" / "references" / "pytest.md"
    ).exists()
    assert (
        settings.brain_root / "library" / "skills" / "codex-skills" / "python-helper" / "scripts" / "check.py"
    ).exists()


def test_brain_service_refresh_archives_stale_working_memory(tmp_path: Path):
    settings = make_settings(tmp_path)
    store = MemoryStore(settings.db_path, settings.brain_root)
    fs_tool = FilesystemTool(settings.repo_root, settings.brain_root)
    llm = StubLLM(
        [
            {
                "remember": [
                    {
                        "category": "objective",
                        "title": "current_project",
                        "content": "Build a self-managing brain workspace.",
                        "tags": ["objective", "brain"],
                        "confidence": 0.9,
                    }
                ],
                "forget_ids": [],
                "workspace_actions": [
                    {
                        "action": "append_file",
                        "path": "attention.md",
                        "content": "- Refresh kept the brain focused.\n",
                    }
                ],
                "skills": [],
            }
        ]
    )
    brain = BrainService(settings, store, llm, fs_tool)

    old_timestamp = "2026-03-20T10:00:00+00:00"
    stale_id = store.store_memory(
        category="experience",
        title="old interaction",
        content="A transient interaction log.",
        tags=["experience"],
        created_at=old_timestamp,
        last_seen_at=old_timestamp,
    )

    result = asyncio.run(brain.refresh(reason="test", force=True))
    archived_files = list((settings.brain_root / "knowledge" / "archive" / "memories").rglob("*.md"))

    assert stale_id in result["archived_ids"]
    assert archived_files
    assert not any(item["id"] == stale_id for item in store.recall(limit=20))
    assert any(item["title"] == "current_project" for item in store.recall("self-managing", limit=10))
    assert "Refresh kept the brain focused." in (settings.brain_workspace_dir / "attention.md").read_text(
        encoding="utf-8"
    )
