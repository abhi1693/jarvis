import asyncio
from pathlib import Path

from app.config import Settings
from app.llm import LLMAdapter, _CodexUnsupportedModelError


def make_settings(
    tmp_path: Path,
    *,
    llm_compat_url: str | None = None,
    llm_model: str | None = None,
    codex_cli_path: str | None = "/usr/bin/codex",
    codex_cli_model: str | None = "gpt-5.1-mini",
) -> Settings:
    data_dir = tmp_path / "data"
    brain_root = data_dir / "agent_brain"
    brain_workspace_dir = brain_root / "workspace"
    brain_skill_dir = brain_workspace_dir / "skills"
    return Settings(
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
        llm_timeout_seconds=90,
        memory_recall_limit=8,
        brain_refresh_interval_seconds=1,
        brain_working_memory_ttl_seconds=60,
        llm_compat_url=llm_compat_url,
        llm_model=llm_model,
        llm_api_key=None,
        codex_cli_path=codex_cli_path,
        codex_cli_model=codex_cli_model,
    )


def test_settings_enable_llm_when_codex_cli_is_available(tmp_path: Path):
    settings = make_settings(tmp_path, codex_cli_path="/usr/bin/codex")

    assert settings.llm_http_enabled is False
    assert settings.codex_cli_enabled is True
    assert settings.llm_enabled is True


def test_llm_adapter_uses_codex_cli_for_text_when_http_is_disabled(monkeypatch, tmp_path: Path):
    settings = make_settings(tmp_path, llm_compat_url=None, llm_model=None)
    adapter = LLMAdapter(settings)
    calls: list[dict[str, object]] = []

    async def fake_run(prompt: str, output_schema=None) -> str:
        calls.append({"prompt": prompt, "output_schema": output_schema})
        return "Codex reply."

    monkeypatch.setattr(adapter, "_run_codex_cli", fake_run)

    result = asyncio.run(
        adapter.complete_text(
            system_prompt="Keep responses direct.",
            user_prompt="Say hello.",
        )
    )

    assert result == "Codex reply."
    assert calls[0]["output_schema"] is None
    assert "Do not inspect the repository, run shell commands, or use tools." in calls[0]["prompt"]
    assert "System prompt:\nKeep responses direct." in calls[0]["prompt"]
    assert "User prompt:\nSay hello." in calls[0]["prompt"]


def test_llm_adapter_falls_back_to_codex_when_http_returns_no_reply(monkeypatch, tmp_path: Path):
    settings = make_settings(tmp_path, llm_compat_url="http://llm.test", llm_model="test-model")
    adapter = LLMAdapter(settings)

    async def fake_http(*_args, **_kwargs):
        return None

    async def fake_codex(prompt: str, output_schema=None) -> str:
        assert output_schema is None
        assert "User prompt:\nNeed a response." in prompt
        return "Codex fallback reply."

    monkeypatch.setattr(adapter, "_complete_http_chat", fake_http)
    monkeypatch.setattr(adapter, "_run_codex_cli", fake_codex)

    result = asyncio.run(
        adapter.complete_text(
            system_prompt="System",
            user_prompt="Need a response.",
        )
    )

    assert result == "Codex fallback reply."


def test_llm_adapter_retries_without_model_when_requested_codex_model_is_unsupported(
    monkeypatch,
    tmp_path: Path,
):
    settings = make_settings(tmp_path, llm_compat_url=None, llm_model=None)
    adapter = LLMAdapter(settings)
    attempted_models: list[str | None] = []

    async def fake_run_once(prompt: str, output_schema, model: str | None) -> str:
        attempted_models.append(model)
        if model == "gpt-5.1-mini":
            raise _CodexUnsupportedModelError("unsupported model")
        assert output_schema is None
        assert prompt == "ping"
        return "pong"

    monkeypatch.setattr(adapter, "_run_codex_cli_once", fake_run_once)

    result = asyncio.run(adapter._run_codex_cli("ping"))

    assert result == "pong"
    assert attempted_models == ["gpt-5.1-mini", None]


def test_llm_adapter_parses_json_from_codex_cli(monkeypatch, tmp_path: Path):
    settings = make_settings(tmp_path, llm_compat_url=None, llm_model=None)
    adapter = LLMAdapter(settings)
    calls: list[dict[str, object]] = []

    async def fake_run(prompt: str, output_schema=None) -> str:
        calls.append({"prompt": prompt, "output_schema": output_schema})
        return '{"name":"conversation","confidence":0.9}'

    monkeypatch.setattr(adapter, "_run_codex_cli", fake_run)

    result = asyncio.run(
        adapter.complete_json(
            system_prompt="Return structured intent JSON.",
            user_prompt="Message: hello",
        )
    )

    assert result == {"name": "conversation", "confidence": 0.9}
    assert calls[0]["output_schema"] is None
    assert "Return only a valid JSON object." in calls[0]["prompt"]
