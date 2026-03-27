from __future__ import annotations

from pathlib import Path
from typing import Any

from app.memory_store import MemoryStore
from app.tools.filesystem import FilesystemTool
from app.tools.shell import ShellTool


class SelfImprovementService:
    def __init__(self, repo_root: Path, memory_store: MemoryStore, fs_tool: FilesystemTool, shell_tool: ShellTool):
        self._repo_root = repo_root
        self._memory_store = memory_store
        self._fs_tool = fs_tool
        self._shell_tool = shell_tool

    def scan(self) -> dict[str, Any]:
        insights: list[dict[str, Any]] = []
        tool_trace: list[dict[str, Any]] = []

        checks = [
            ("Compile Python package", "python3 -m compileall app"),
            ("Run tests", "pytest -q"),
            ("Run Ruff", "ruff check ."),
        ]

        for label, command in checks:
            result = self._shell_tool.run(command)
            tool_trace.append({"tool": "run_command", "label": label, "command": command, "result": result})
            insights.extend(self._insights_from_command(label, command, result))

        todo_result = self._scan_todo_comments()
        tool_trace.append({"tool": "run_command", "label": "Search TODO comments", "result": todo_result})
        if todo_result.get("ok") and todo_result.get("matches"):
            insights.append(
                {
                    "severity": "low",
                    "source": "repo_scan",
                    "title": "Open TODO markers",
                    "details": f"Found {len(todo_result['matches'])} TODO markers in the repository.",
                    "file_path": todo_result["matches"][0]["path"],
                    "line_number": todo_result["matches"][0]["line_number"],
                }
            )

        tests_dir = self._repo_root / "tests"
        if not tests_dir.exists():
            insights.append(
                {
                    "severity": "medium",
                    "source": "repo_scan",
                    "title": "No tests directory found",
                    "details": "Create tests to give the system a safer evolution loop.",
                    "file_path": "tests",
                    "line_number": None,
                }
            )

        persisted = [self._persist_insight(insight) for insight in insights]
        return {
            "summary": self._build_summary(insights),
            "insights": persisted,
            "tool_trace": tool_trace,
        }

    def format_repo(self) -> dict[str, Any]:
        result = self._shell_tool.run("ruff format .")
        return {"summary": "Formatting run completed." if result.get("ok") else "Formatting failed.", "result": result}

    def _insights_from_command(self, label: str, command: str, result: dict[str, Any]) -> list[dict[str, Any]]:
        if result.get("ok"):
            return []

        error = result.get("error")
        details = error or result.get("stderr") or result.get("stdout") or "Command failed without output."
        severity = "high" if "compileall" in command or "pytest" in command else "medium"
        return [
            {
                "severity": severity,
                "source": command,
                "title": label,
                "details": details[:4000],
                "file_path": None,
                "line_number": None,
            }
        ]

    def _persist_insight(self, insight: dict[str, Any]) -> dict[str, Any]:
        record_id = self._memory_store.store_evolution_insight(
            severity=insight["severity"],
            source=insight["source"],
            title=insight["title"],
            details=insight["details"],
            file_path=insight.get("file_path"),
            line_number=insight.get("line_number"),
        )
        insight["id"] = record_id
        insight["status"] = "open"
        return insight

    def _build_summary(self, insights: list[dict[str, Any]]) -> str:
        if not insights:
            return "Evolution scan completed with no new insights."
        counts = {
            severity: sum(1 for insight in insights if insight["severity"] == severity)
            for severity in ("high", "medium", "low")
        }
        return (
            "Evolution scan completed. "
            f"High: {counts['high']}, medium: {counts['medium']}, low: {counts['low']} insights recorded."
        )

    def _scan_todo_comments(self) -> dict[str, Any]:
        targets = [path for path in ("app", "tests", "README.md") if (self._repo_root / path).exists()]
        if not targets:
            return {"ok": True, "matches": []}

        command = 'rg -n "^(\\s*)(#|//|/\\*|<!--)\\s*TODO:" ' + " ".join(targets)
        result = self._shell_tool.run(command)

        if result.get("ok") or result.get("returncode") == 1:
            matches = []
            for line in result.get("stdout", "").splitlines():
                parts = line.split(":", maxsplit=2)
                if len(parts) != 3:
                    continue
                matches.append(
                    {
                        "path": parts[0],
                        "line_number": int(parts[1]),
                        "line": parts[2].strip(),
                    }
                )
            return {"ok": True, "matches": matches}

        return {"ok": False, "error": result.get("error") or result.get("stderr") or "rg failed", "matches": []}
