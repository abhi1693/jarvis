from __future__ import annotations

from pathlib import Path
from typing import Any


class FilesystemTool:
    _SKIP_PARTS = {".git", ".venv", "__pycache__", "node_modules"}
    _SKIP_SUFFIXES = {".db", ".png", ".jpg", ".jpeg", ".gif", ".pyc", ".npy"}

    def __init__(self, repo_root: Path, brain_root: Path | None = None):
        self._repo_root = repo_root.resolve()
        self._brain_root = (
            brain_root.resolve() if brain_root else (self._repo_root / "data" / "agent_brain").resolve()
        )

    def list_directory(self, path: str = ".") -> dict[str, Any]:
        target = self._resolve(path)
        if not target.exists():
            return {"ok": False, "error": f"{path} does not exist"}
        if not target.is_dir():
            return {"ok": False, "error": f"{path} is not a directory"}

        items = []
        for item in sorted(target.iterdir(), key=lambda current: (current.is_file(), current.name.lower())):
            if self._should_skip_path(item):
                continue
            items.append(
                {
                    "name": item.name,
                    "path": str(item.relative_to(self._repo_root)),
                    "type": "dir" if item.is_dir() else "file",
                }
            )

        return {"ok": True, "path": str(target.relative_to(self._repo_root)), "items": items}

    def read_file(self, path: str, max_chars: int = 12_000) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.exists():
            return {"ok": False, "error": f"{path} does not exist"}
        if not target.is_file():
            return {"ok": False, "error": f"{path} is not a file"}

        content = target.read_text(encoding="utf-8")
        truncated = len(content) > max_chars
        if truncated:
            content = content[:max_chars]

        return {
            "ok": True,
            "path": str(target.relative_to(self._repo_root)),
            "content": content,
            "truncated": truncated,
        }

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"ok": True, "path": str(target.relative_to(self._repo_root)), "bytes_written": len(content)}

    def make_directory(self, path: str) -> dict[str, Any]:
        target = self._resolve(path)
        target.mkdir(parents=True, exist_ok=True)
        return {"ok": True, "path": str(target.relative_to(self._repo_root))}

    def move_path(self, source: str, destination: str) -> dict[str, Any]:
        source_path = self._resolve(source)
        destination_path = self._resolve(destination)
        if not source_path.exists():
            return {"ok": False, "error": f"{source} does not exist"}
        if destination_path.exists():
            return {"ok": False, "error": f"{destination} already exists"}

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.rename(destination_path)
        return {
            "ok": True,
            "source": str(source_path.relative_to(self._repo_root)),
            "destination": str(destination_path.relative_to(self._repo_root)),
        }

    def search_text(self, pattern: str, path: str = ".", max_matches: int = 25) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.exists():
            return {"ok": False, "error": f"{path} does not exist"}

        matches = []
        search_paths = [target] if target.is_file() else list(target.rglob("*"))
        lowered = pattern.lower()

        for candidate in search_paths:
            if not candidate.is_file():
                continue
            if self._should_skip_path(candidate):
                continue

            try:
                content = candidate.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

            for line_number, line in enumerate(content.splitlines(), start=1):
                if lowered in line.lower():
                    matches.append(
                        {
                            "path": str(candidate.relative_to(self._repo_root)),
                            "line_number": line_number,
                            "line": line.strip(),
                        }
                    )
                    if len(matches) >= max_matches:
                        return {"ok": True, "matches": matches, "truncated": True}

        return {"ok": True, "matches": matches, "truncated": False}

    def _resolve(self, path: str) -> Path:
        candidate = (self._repo_root / path).resolve()
        if self._repo_root not in candidate.parents and candidate != self._repo_root:
            raise ValueError(f"Path escapes repo root: {path}")
        return candidate

    def _should_skip_path(self, candidate: Path) -> bool:
        try:
            relative = candidate.relative_to(self._repo_root)
        except ValueError:
            return True

        if any(part in self._SKIP_PARTS for part in relative.parts):
            return True

        if relative.parts and relative.parts[0] == "data" and not self._is_brain_path(candidate):
            return True

        if candidate.is_file() and candidate.suffix.lower() in self._SKIP_SUFFIXES:
            return True

        return False

    def _is_brain_path(self, candidate: Path) -> bool:
        return candidate == self._brain_root or self._brain_root in candidate.parents
