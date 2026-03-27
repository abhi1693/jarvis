from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    app_name: str
    repo_root: Path
    data_dir: Path
    db_path: Path
    brain_root: Path
    brain_workspace_dir: Path
    brain_skill_dir: Path
    brain_skill_source_dirs: tuple[Path, ...]
    admin_face_path: Path
    snapshot_dir: Path
    media_dir: Path
    change_set_dir: Path
    command_timeout_seconds: int
    llm_timeout_seconds: int
    memory_recall_limit: int
    brain_refresh_interval_seconds: int
    brain_working_memory_ttl_seconds: int
    llm_compat_url: str | None
    llm_model: str | None
    llm_api_key: str | None
    codex_cli_path: str | None
    codex_cli_model: str | None

    @property
    def llm_http_enabled(self) -> bool:
        return bool(self.llm_compat_url and self.llm_model)

    @property
    def codex_cli_enabled(self) -> bool:
        return bool(self.codex_cli_path)

    @property
    def llm_enabled(self) -> bool:
        return self.llm_http_enabled or self.codex_cli_enabled

    @property
    def static_dir(self) -> Path:
        return self.repo_root / "app" / "static"


def get_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    brain_root = data_dir / "agent_brain"
    brain_workspace_dir = brain_root / "workspace"
    brain_skill_dir = brain_workspace_dir / "skills"
    raw_skill_sources = os.getenv("JARVIS_BRAIN_SKILL_SOURCES", "~/.codex/skills,~/agents/skills")
    brain_skill_source_dirs = tuple(
        Path(item.strip()).expanduser()
        for item in raw_skill_sources.split(",")
        if item.strip()
    )
    settings = Settings(
        app_name=os.getenv("AGENT_NAME", "Adaptive Agent"),
        repo_root=repo_root,
        data_dir=data_dir,
        db_path=data_dir / "jarvis.db",
        brain_root=brain_root,
        brain_workspace_dir=brain_workspace_dir,
        brain_skill_dir=brain_skill_dir,
        brain_skill_source_dirs=brain_skill_source_dirs,
        admin_face_path=data_dir / "admin_face.npy",
        snapshot_dir=data_dir / "snapshots",
        media_dir=data_dir / "media",
        change_set_dir=data_dir / "change_sets",
        command_timeout_seconds=int(os.getenv("JARVIS_COMMAND_TIMEOUT", "20")),
        llm_timeout_seconds=int(os.getenv("JARVIS_LLM_TIMEOUT", "90")),
        memory_recall_limit=int(os.getenv("JARVIS_MEMORY_RECALL_LIMIT", "8")),
        brain_refresh_interval_seconds=int(os.getenv("JARVIS_BRAIN_REFRESH_INTERVAL", "90")),
        brain_working_memory_ttl_seconds=int(os.getenv("JARVIS_BRAIN_WORKING_TTL", "21600")),
        llm_compat_url=os.getenv("LLM_COMPAT_URL"),
        llm_model=os.getenv("LLM_MODEL"),
        llm_api_key=os.getenv("LLM_API_KEY"),
        codex_cli_path=os.getenv("CODEX_CLI_PATH") or shutil.which("codex"),
        codex_cli_model=os.getenv("CODEX_CLI_MODEL", "gpt-5.1-mini").strip() or None,
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.brain_root.mkdir(parents=True, exist_ok=True)
    settings.brain_workspace_dir.mkdir(parents=True, exist_ok=True)
    settings.brain_skill_dir.mkdir(parents=True, exist_ok=True)
    settings.snapshot_dir.mkdir(parents=True, exist_ok=True)
    settings.media_dir.mkdir(parents=True, exist_ok=True)
    settings.change_set_dir.mkdir(parents=True, exist_ok=True)
    return settings
