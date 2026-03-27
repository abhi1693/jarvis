from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    app_name: str
    repo_root: Path
    data_dir: Path
    db_path: Path
    admin_face_path: Path
    snapshot_dir: Path
    media_dir: Path
    change_set_dir: Path
    command_timeout_seconds: int
    memory_recall_limit: int
    llm_compat_url: str | None
    llm_model: str | None
    llm_api_key: str | None

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_compat_url and self.llm_model)

    @property
    def static_dir(self) -> Path:
        return self.repo_root / "app" / "static"


def get_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    settings = Settings(
        app_name="Jarvis",
        repo_root=repo_root,
        data_dir=data_dir,
        db_path=data_dir / "jarvis.db",
        admin_face_path=data_dir / "admin_face.npy",
        snapshot_dir=data_dir / "snapshots",
        media_dir=data_dir / "media",
        change_set_dir=data_dir / "change_sets",
        command_timeout_seconds=int(os.getenv("JARVIS_COMMAND_TIMEOUT", "20")),
        memory_recall_limit=int(os.getenv("JARVIS_MEMORY_RECALL_LIMIT", "8")),
        llm_compat_url=os.getenv("LLM_COMPAT_URL"),
        llm_model=os.getenv("LLM_MODEL"),
        llm_api_key=os.getenv("LLM_API_KEY"),
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.snapshot_dir.mkdir(parents=True, exist_ok=True)
    settings.media_dir.mkdir(parents=True, exist_ok=True)
    settings.change_set_dir.mkdir(parents=True, exist_ok=True)
    return settings
