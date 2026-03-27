import asyncio
from pathlib import Path

from app.agent import AgentRuntime
from app.brain import BrainService
from app.config import Settings
from app.memory_store import MemoryStore
from app.models import IntentResult
from app.tools.filesystem import FilesystemTool


class CapturingLLM:
    def __init__(self) -> None:
        self.text_calls: list[dict[str, str]] = []

    @property
    def enabled(self) -> bool:
        return True

    async def complete_json(self, _system_prompt: str, _user_prompt: str):
        return None

    async def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        self.text_calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return "Skill-aware reply."


class StubIntentService:
    async def parse(self, _message: str) -> IntentResult:
        return IntentResult(
            name="create",
            confidence=0.9,
            suggested_tools=["search_text", "read_file"],
            memory_candidates=[],
        )


class StubConversationIntentService:
    async def parse(self, _message: str) -> IntentResult:
        return IntentResult(
            name="conversation",
            confidence=0.9,
            suggested_tools=[],
            memory_candidates=[],
        )


class DisabledLLM:
    @property
    def enabled(self) -> bool:
        return False

    async def complete_json(self, _system_prompt: str, _user_prompt: str):
        return None

    async def complete_text(self, _system_prompt: str, _user_prompt: str) -> str | None:
        return None


class StubShellTool:
    def run(self, _command: str) -> dict[str, object]:
        return {"ok": True, "returncode": 0, "stdout": "", "stderr": ""}


class StubWebSearchTool:
    def search(self, _query: str, max_results: int = 5) -> dict[str, object]:
        return {"ok": True, "results": [], "max_results": max_results}


class StubSelfImprovementService:
    def format_repo(self) -> dict[str, object]:
        return {"summary": "formatted", "result": {"ok": True, "returncode": 0, "stdout": "", "stderr": ""}}

    def scan(self) -> dict[str, object]:
        return {"summary": "scanned", "tool_trace": [], "insights": []}


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


def test_agent_runtime_discovers_skill_bundles_before_prompting_llm(tmp_path: Path):
    settings = make_settings(tmp_path)
    external_root = tmp_path / ".codex" / "skills"
    external_skill = external_root / "python-helper" / "SKILL.md"
    external_skill.parent.mkdir(parents=True, exist_ok=True)
    (external_skill.parent / "references").mkdir(parents=True, exist_ok=True)
    external_skill.write_text(
        "# Python Helper\n\nUse this skill when debugging Python services and test failures.\n",
        encoding="utf-8",
    )
    (external_skill.parent / "references" / "pytest.md").write_text(
        "Prefer pytest fixtures and assertion introspection when debugging tests.\n",
        encoding="utf-8",
    )
    settings.brain_skill_source_dirs = (external_root,)

    llm = CapturingLLM()
    store = MemoryStore(settings.db_path, settings.brain_root, settings.brain_skill_source_dirs)
    fs_tool = FilesystemTool(settings.repo_root, settings.brain_root)
    brain = BrainService(settings, store, llm, fs_tool)
    agent = AgentRuntime(
        settings,
        store,
        StubIntentService(),
        llm,
        fs_tool,
        StubShellTool(),
        StubWebSearchTool(),
        StubSelfImprovementService(),
        brain,
    )

    response = asyncio.run(
        agent.handle_interaction("Help me debug the Python tests in this repo.", modality="text")
    )

    assert response.message == "Skill-aware reply."
    assert llm.text_calls
    prompt = llm.text_calls[-1]["user_prompt"]
    system_prompt = llm.text_calls[-1]["system_prompt"]
    assert "Selected skill files for this turn:" in prompt
    assert "python-helper" in prompt
    assert "Use this skill when debugging Python services and test failures." in prompt
    assert "Prefer pytest fixtures and assertion introspection when debugging tests." in prompt
    assert "Selected skill files in the operating context are active instructions for this turn." in system_prompt


def test_agent_runtime_fallback_conversation_varies_without_llm(tmp_path: Path):
    settings = make_settings(tmp_path)
    settings.llm_compat_url = None
    settings.llm_model = None
    llm = DisabledLLM()
    store = MemoryStore(settings.db_path, settings.brain_root, settings.brain_skill_source_dirs)
    fs_tool = FilesystemTool(settings.repo_root, settings.brain_root)
    brain = BrainService(settings, store, llm, fs_tool)
    agent = AgentRuntime(
        settings,
        store,
        StubConversationIntentService(),
        llm,
        fs_tool,
        StubShellTool(),
        StubWebSearchTool(),
        StubSelfImprovementService(),
        brain,
    )

    greeting = asyncio.run(agent.handle_interaction("hello", modality="text"))
    capabilities = asyncio.run(agent.handle_interaction("what can you do?", modality="text"))

    assert greeting.message == "Hello. I’m here."
    assert "read and write files" in capabilities.message
    assert "Open-ended chat is limited" in capabilities.message
    assert capabilities.message != greeting.message
