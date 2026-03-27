from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any


class ShellTool:
    _ALLOWED_COMMANDS = {
        "git",
        "ls",
        "node",
        "npm",
        "pwd",
        "py.test",
        "pytest",
        "python",
        "python3",
        "rg",
        "ruff",
        "sed",
        "uvicorn",
    }
    _BLOCKED_TOKENS = {"rm", "shutdown", "reboot", "mkfs", "dd", "sudo", "chmod", "chown"}

    def __init__(self, repo_root: Path, timeout_seconds: int):
        self._repo_root = repo_root
        self._timeout_seconds = timeout_seconds

    def run(self, command: str) -> dict[str, Any]:
        tokens = shlex.split(command)
        if not tokens:
            return {"ok": False, "error": "No command provided"}

        program = tokens[0]
        if program in self._BLOCKED_TOKENS or any(token in self._BLOCKED_TOKENS for token in tokens):
            return {"ok": False, "error": "Command blocked by policy"}
        if program not in self._ALLOWED_COMMANDS:
            return {"ok": False, "error": f"Command '{program}' is not allowed"}
        if shutil.which(program) is None:
            return {"ok": False, "error": f"Command '{program}' is not installed"}

        try:
            completed = subprocess.run(
                tokens,
                cwd=self._repo_root,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"Command timed out after {self._timeout_seconds}s"}

        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }

