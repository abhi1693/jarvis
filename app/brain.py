from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from app.config import Settings
from app.llm import LLMAdapter
from app.memory_store import MemoryStore
from app.models import MemoryCandidate
from app.tools.filesystem import FilesystemTool


class BrainService:
    def __init__(
        self,
        settings: Settings,
        memory_store: MemoryStore,
        llm_adapter: LLMAdapter,
        fs_tool: FilesystemTool,
    ):
        self._settings = settings
        self._memory_store = memory_store
        self._llm_adapter = llm_adapter
        self._fs_tool = fs_tool
        self._refresh_lock = asyncio.Lock()
        self._last_refresh_started_at = 0.0

    def build_prompt_context(
        self,
        query: str,
        runtime_context: list[dict[str, Any]],
        recalled_memories: list[dict[str, Any]],
        skills: list[dict[str, Any]],
        discovered_skills: list[dict[str, Any]] | None = None,
    ) -> str:
        brain = self._memory_store.get_brain_snapshot(
            query=query,
            memory_limit=self._settings.memory_recall_limit,
            interaction_limit=6,
            workspace_limit=4,
            skill_limit=4,
        )
        sections = []

        if runtime_context:
            sections.append(
                "Current operating context:\n"
                + "\n".join(
                    f"- [{memory['category']}] {memory['title']}: {memory['content']}"
                    for memory in runtime_context[:8]
                )
            )
        elif recalled_memories:
            sections.append(
                "Relevant memories:\n"
                + "\n".join(
                    f"- [{memory['category']}] {memory['title']}: {memory['content']}"
                    for memory in recalled_memories[:8]
                )
            )

        active_skill_notes = []
        for item in discovered_skills or []:
            note = (
                f"- {item['name']} [{item['source']}] root={item['root_path']} main={item['main_path']}\n"
                f"{item['content']}"
            )
            support_files = item.get("support_files", [])
            if support_files:
                rendered_support = "\n".join(
                    f"  - {file_item['path']}\n{file_item['content']}" for file_item in support_files
                )
                note += f"\nSupporting skill files:\n{rendered_support}"
            active_skill_notes.append(note)
        if active_skill_notes:
            sections.append(
                "Selected skill files for this turn:\n"
                + "\n".join(active_skill_notes)
            )

        brain_sections = []
        for label, key in (
            ("Persona file", "persona"),
            ("Working memory file", "working_memory"),
            ("Long-term memory file", "long_term_memory"),
            ("Skills file", "skills"),
            ("Skill library", "skill_library"),
        ):
            content = str(brain.get(key, "")).strip()
            if content:
                brain_sections.append(f"{label}:\n{content}")
        if brain_sections:
            sections.append("\n\n".join(brain_sections))

        workspace_notes = []
        for item in brain.get("workspace_files", []):
            workspace_notes.append(f"- {item['path']}\n{item['content']}")
        if workspace_notes:
            sections.append("Workspace notes:\n" + "\n".join(workspace_notes))

        skill_reference_notes = []
        for item in brain.get("skill_reference_files", []):
            skill_reference_notes.append(
                f"- {item['name']} [{item['source']}] {item['path']}\n{item['content']}"
            )
        if skill_reference_notes:
            sections.append("Relevant skill references:\n" + "\n".join(skill_reference_notes))

        recent_skills = []
        for skill in skills[:6]:
            recent_skills.append(f"- {skill['name']} ({skill['trigger_hint']}): {skill['description']}")
        if recent_skills:
            sections.append("Learned skills:\n" + "\n".join(recent_skills))

        recent_interactions = []
        for interaction in brain.get("recent_interactions", []):
            recent_interactions.append(f"- {interaction['role']}/{interaction['intent']}: {interaction['content'][:180]}")
        if recent_interactions:
            sections.append("Recent interaction trail:\n" + "\n".join(recent_interactions))

        return "\n\n".join(section for section in sections if section).strip() or "- no durable operating context yet"

    def discover_skill_bundles(
        self,
        query: str,
        *,
        intent_name: str = "",
        suggested_tools: list[str] | None = None,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        return self._memory_store.discover_skill_bundles(
            query,
            intent_name=intent_name,
            suggested_tools=suggested_tools or [],
            limit=limit,
        )

    async def ingest_user_message(
        self,
        *,
        message: str,
        modality: str,
        intent_name: str,
        memory_candidates: list[MemoryCandidate],
    ) -> dict[str, Any]:
        plan = await self._plan_user_memory(message, modality, intent_name, memory_candidates)
        remembered = self._persist_memories(plan.get("remember", []), modality)
        forgotten = self._forget_memories(plan.get("forget_ids", []))
        actions = self._apply_workspace_actions(plan.get("workspace_actions", []))
        self._memory_store.refresh_brain_documents()
        return {
            "remembered": remembered,
            "forgotten_ids": forgotten,
            "workspace_actions": actions,
        }

    def learn_from_tool_trace(self, intent_name: str, tool_trace: list[dict[str, Any]], message: str) -> None:
        if not tool_trace:
            return

        steps = []
        for item in tool_trace:
            steps.append(
                {
                    "tool": item.get("tool"),
                    "label": item.get("label"),
                    "pattern": item.get("pattern"),
                    "command": item.get("command"),
                    "query": item.get("query"),
                }
            )

        self._memory_store.store_skill(
            name=f"{intent_name.replace('_', ' ').title()} skill",
            description=f"A learned execution pattern captured from: {message[:120]}",
            trigger_hint=intent_name,
            steps=steps,
        )

        timestamp = self._memory_store.current_timestamp()
        self._apply_workspace_actions(
            [
                {
                    "action": "append_file",
                    "path": "skill-journal.md",
                    "content": (
                        f"\n## {timestamp}\n"
                        f"- intent: {intent_name}\n"
                        f"- trigger: {message[:180]}\n"
                        f"- tools: {', '.join(step.get('tool') or 'unknown' for step in steps)}\n"
                    ),
                }
            ]
        )
        self._memory_store.refresh_brain_documents()

    async def refresh(self, *, reason: str = "periodic", force: bool = False) -> dict[str, Any]:
        interval = max(self._settings.brain_refresh_interval_seconds, 1)
        if not force and (time.monotonic() - self._last_refresh_started_at) < interval:
            return {"ok": True, "skipped": True, "reason": reason}

        async with self._refresh_lock:
            if not force and (time.monotonic() - self._last_refresh_started_at) < interval:
                return {"ok": True, "skipped": True, "reason": reason}

            self._last_refresh_started_at = time.monotonic()
            archived_ids = self._memory_store.archive_stale_memories(
                categories={"context", "experience", "note", "state"},
                older_than_seconds=self._settings.brain_working_memory_ttl_seconds,
                limit=24,
            )
            plan = await self._plan_refresh(reason, archived_ids)
            remembered = self._persist_memories(plan.get("remember", []), source="brain_refresh")
            forgotten = self._forget_memories(plan.get("forget_ids", []))
            skills = self._persist_skills(plan.get("skills", []))
            actions = self._apply_workspace_actions(plan.get("workspace_actions", []))

            refresh_log = (
                f"\n## {self._memory_store.current_timestamp()}\n"
                f"- reason: {reason}\n"
                f"- archived stale working memories: {len(archived_ids)}\n"
                f"- remembered: {len(remembered)}\n"
                f"- forgot: {len(forgotten)}\n"
                f"- skills updated: {len(skills)}\n"
                f"- workspace actions: {len(actions)}\n"
            )
            self._apply_workspace_actions(
                [{"action": "append_file", "path": "refresh-log.md", "content": refresh_log}]
            )
            self._memory_store.refresh_brain_documents()
            return {
                "ok": True,
                "skipped": False,
                "reason": reason,
                "archived_ids": archived_ids,
                "remembered": remembered,
                "forgotten_ids": forgotten,
                "skills": skills,
                "workspace_actions": actions,
            }

    async def _plan_user_memory(
        self,
        message: str,
        modality: str,
        intent_name: str,
        memory_candidates: list[MemoryCandidate],
    ) -> dict[str, Any]:
        if self._llm_adapter.enabled:
            snapshot = self._memory_store.get_brain_snapshot(
                query=message,
                memory_limit=self._settings.memory_recall_limit,
                interaction_limit=6,
                workspace_limit=4,
            )
            system_prompt = """
You manage memory for a local adaptive agent with a markdown brain and a freeform workspace.
Return JSON with:
- remember: array of memory objects {category,title,content,tags,confidence}
- forget_ids: array of memory ids to archive if they are clearly stale or superseded
- workspace_actions: array of actions using only paths relative to the workspace root

Rules:
- Remember only durable facts, operator preferences, duties, constraints, ongoing work state, or concise actionable notes.
- Do not store routine chit-chat, acknowledgements, or full transcripts unless they carry lasting value.
- Normalize memories into concise statements.
- Workspace actions may use only: write_file, append_file, make_directory, move_path, delete_path.
- Workspace actions must stay inside the workspace root and use relative paths like plans/current.md.
- If nothing should be remembered, return empty arrays.
Only return JSON.
""".strip()
            user_prompt = (
                f"Intent: {intent_name}\n"
                f"Modality: {modality}\n"
                f"Message: {message}\n"
                f"Extracted candidates: {[candidate.model_dump() for candidate in memory_candidates]}\n"
                f"Brain snapshot: {snapshot}"
            )
            try:
                response = await self._llm_adapter.complete_json(system_prompt, user_prompt)
            except Exception:
                response = None
            if response:
                return {
                    "remember": self._coerce_memory_items(response.get("remember", [])),
                    "forget_ids": self._coerce_ids(response.get("forget_ids", [])),
                    "workspace_actions": self._coerce_workspace_actions(response.get("workspace_actions", [])),
                }

        return self._fallback_user_memory_plan(message, modality, intent_name, memory_candidates)

    async def _plan_refresh(self, reason: str, archived_ids: list[int]) -> dict[str, Any]:
        if not self._llm_adapter.enabled:
            return {"remember": [], "forget_ids": [], "workspace_actions": [], "skills": []}

        snapshot = self._memory_store.get_brain_snapshot(
            query="",
            memory_limit=12,
            interaction_limit=10,
            workspace_limit=6,
        )
        system_prompt = """
You periodically refresh a local agent brain.
Return JSON with:
- remember: array of distilled memory objects {category,title,content,tags,confidence}
- forget_ids: array of stale memory ids to archive
- skills: array of learned skill objects {name,description,trigger_hint,steps}
- workspace_actions: array of workspace actions using only relative paths

Rules:
- Consolidate recent interactions into fewer, better memories.
- Prefer promoting stable preferences, duties, and active projects over storing raw chat text.
- Forget only stale working memory or superseded notes.
- Keep workspace actions inside the workspace root.
- If no change is needed, return empty arrays.
Only return JSON.
""".strip()
        user_prompt = (
            f"Refresh reason: {reason}\n"
            f"Already archived this cycle: {archived_ids}\n"
            f"Brain snapshot: {snapshot}"
        )
        try:
            response = await self._llm_adapter.complete_json(system_prompt, user_prompt)
        except Exception:
            response = None
        if not response:
            return {"remember": [], "forget_ids": [], "workspace_actions": [], "skills": []}

        return {
            "remember": self._coerce_memory_items(response.get("remember", [])),
            "forget_ids": self._coerce_ids(response.get("forget_ids", [])),
            "workspace_actions": self._coerce_workspace_actions(response.get("workspace_actions", [])),
            "skills": self._coerce_skill_items(response.get("skills", [])),
        }

    def _fallback_user_memory_plan(
        self,
        message: str,
        modality: str,
        intent_name: str,
        memory_candidates: list[MemoryCandidate],
    ) -> dict[str, Any]:
        remember = []
        if intent_name in {"remember", "orient"} and memory_candidates:
            remember = [candidate.model_dump() for candidate in memory_candidates[:8]]
        elif intent_name == "orient":
            remember = [
                {
                    "category": "context",
                    "title": "operator_direction",
                    "content": message,
                    "tags": ["context", "operator", modality],
                    "confidence": 0.82,
                }
            ]
        elif intent_name == "remember":
            remember = [
                {
                    "category": "note",
                    "title": f"{modality} note",
                    "content": message,
                    "tags": ["note", modality],
                    "confidence": 0.72,
                }
            ]
        elif memory_candidates:
            remember = [
                candidate.model_dump()
                for candidate in memory_candidates[:4]
                if candidate.confidence >= 0.8
            ]

        return {"remember": remember, "forget_ids": [], "workspace_actions": []}

    def _persist_memories(self, items: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
        remembered = []
        for item in items:
            content = str(item.get("content", "")).strip()
            title = str(item.get("title", "")).strip() or "note"
            category = str(item.get("category", "")).strip() or "note"
            if not content:
                continue
            record_id = self._memory_store.store_memory(
                category=category,
                title=title,
                content=content,
                tags=[str(tag) for tag in item.get("tags", [])],
                source=source,
                confidence=float(item.get("confidence", 0.78)),
            )
            remembered.append(
                {
                    "id": record_id,
                    "category": category,
                    "title": title,
                    "content": content,
                }
            )
        return remembered

    def _persist_skills(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        stored = []
        for item in items:
            name = str(item.get("name", "")).strip()
            description = str(item.get("description", "")).strip()
            trigger_hint = str(item.get("trigger_hint", "")).strip()
            steps = item.get("steps", [])
            if not name or not description or not trigger_hint:
                continue
            record_id = self._memory_store.store_skill(
                name=name,
                description=description,
                trigger_hint=trigger_hint,
                steps=steps if isinstance(steps, list) else [],
            )
            stored.append({"id": record_id, "name": name, "trigger_hint": trigger_hint})
        return stored

    def _forget_memories(self, memory_ids: list[int]) -> list[int]:
        forgotten = []
        for memory_id in memory_ids:
            if self._memory_store.archive_memory(memory_id, reason="brain_refresh"):
                forgotten.append(memory_id)
        return forgotten

    def _apply_workspace_actions(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        applied = []
        for action in actions:
            action_name = action.get("action")
            try:
                if action_name in {"write_file", "append_file", "make_directory", "delete_path"}:
                    path = self._workspace_repo_path(str(action.get("path", "")).strip())
                    if action_name == "write_file":
                        result = self._fs_tool.write_file(path, str(action.get("content", "")))
                    elif action_name == "append_file":
                        result = self._fs_tool.append_file(path, str(action.get("content", "")))
                    elif action_name == "make_directory":
                        result = self._fs_tool.make_directory(path)
                    else:
                        result = self._fs_tool.delete_path(path, recursive=bool(action.get("recursive", False)))
                elif action_name == "move_path":
                    source = self._workspace_repo_path(str(action.get("source", "")).strip())
                    destination = self._workspace_repo_path(str(action.get("destination", "")).strip())
                    result = self._fs_tool.move_path(source, destination)
                else:
                    continue
            except ValueError:
                continue

            if result.get("ok"):
                applied.append({"action": action_name, "result": result})
        return applied

    def _workspace_repo_path(self, relative_path: str) -> str:
        candidate = Path(relative_path)
        if not relative_path:
            raise ValueError("workspace path required")
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("workspace paths must be relative")
        workspace_root = self._settings.brain_workspace_dir.relative_to(self._settings.repo_root)
        return str(workspace_root / candidate)

    def _coerce_memory_items(self, items: Any) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []

        remembered = []
        for item in items:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            remembered.append(
                {
                    "category": str(item.get("category", "note")),
                    "title": str(item.get("title", "note")),
                    "content": content,
                    "tags": [str(tag) for tag in item.get("tags", []) if str(tag).strip()],
                    "confidence": float(item.get("confidence", 0.78)),
                }
            )
        return remembered

    def _coerce_workspace_actions(self, actions: Any) -> list[dict[str, Any]]:
        if not isinstance(actions, list):
            return []

        normalized = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            action_name = str(action.get("action", "")).strip()
            if action_name not in {"write_file", "append_file", "make_directory", "move_path", "delete_path"}:
                continue
            normalized.append(action)
        return normalized

    def _coerce_skill_items(self, items: Any) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []
        normalized = []
        for item in items:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "name": str(item.get("name", "")).strip(),
                    "description": str(item.get("description", "")).strip(),
                    "trigger_hint": str(item.get("trigger_hint", "")).strip(),
                    "steps": item.get("steps", []),
                }
            )
        return normalized

    def _coerce_ids(self, values: Any) -> list[int]:
        if not isinstance(values, list):
            return []
        ids = []
        for value in values:
            try:
                ids.append(int(value))
            except (TypeError, ValueError):
                continue
        return ids
