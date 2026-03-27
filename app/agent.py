from __future__ import annotations

import re
from typing import Any

from app.config import Settings
from app.intents import IntentService
from app.llm import LLMAdapter
from app.memory_store import MemoryStore
from app.models import InteractionResponse, IntentResult
from app.self_improvement import SelfImprovementService
from app.tools.filesystem import FilesystemTool
from app.tools.shell import ShellTool
from app.tools.web_search import WebSearchTool


class AgentRuntime:
    def __init__(
        self,
        settings: Settings,
        memory_store: MemoryStore,
        intent_service: IntentService,
        llm_adapter: LLMAdapter,
        fs_tool: FilesystemTool,
        shell_tool: ShellTool,
        web_search_tool: WebSearchTool,
        self_improvement: SelfImprovementService,
    ):
        self._settings = settings
        self._memory_store = memory_store
        self._intent_service = intent_service
        self._llm_adapter = llm_adapter
        self._fs_tool = fs_tool
        self._shell_tool = shell_tool
        self._web_search_tool = web_search_tool
        self._self_improvement = self_improvement

    async def handle_interaction(
        self,
        message: str,
        *,
        modality: str = "text",
        channel: str = "ui",
        metadata: dict[str, Any] | None = None,
        media_path: str | None = None,
    ) -> InteractionResponse:
        normalized_message = message.strip() or f"{modality} interaction"
        intent = await self._intent_service.parse(normalized_message)
        self._memory_store.record_interaction(
            "user",
            normalized_message,
            intent.name,
            modality=modality,
            channel=channel,
            metadata=metadata,
            media_path=media_path,
        )
        self._remember_facts(intent.extracted_facts, modality)
        self._store_experience(normalized_message, modality, intent.name)

        recalled_memories = self._memory_store.recall(normalized_message, limit=self._settings.memory_recall_limit)
        skills = self._memory_store.list_recent_skills(limit=4)

        insights: list[dict[str, Any]] = []
        tool_trace: list[dict[str, Any]] = []

        if intent.name == "remember":
            response_text = self._handle_remember(normalized_message, intent, modality)
        elif intent.name == "memory_query":
            response_text = self._handle_memory_query()
        elif intent.name == "web_search":
            response_text, tool_trace = self._handle_web_search(normalized_message)
        elif intent.name == "tool_use":
            response_text, tool_trace = self._handle_tool_use(normalized_message)
        elif intent.name == "create":
            response_text, tool_trace = await self._handle_creation_request(normalized_message, recalled_memories)
        elif intent.name == "evolve":
            response_text, tool_trace, insights = self._handle_evolution(normalized_message)
        else:
            response_text = await self._handle_conversation(normalized_message, recalled_memories, modality)

        self._memory_store.record_interaction(
            "assistant",
            response_text,
            intent.name,
            modality="text",
            channel=channel,
            metadata={"reply_to": modality},
        )
        self._learn_skill(intent.name, tool_trace, normalized_message)

        return InteractionResponse(
            message=response_text,
            intent=intent,
            memories=recalled_memories[:5],
            skills=skills,
            tool_trace=tool_trace,
            insights=insights,
        )

    def invoke_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "list_directory":
            return self._fs_tool.list_directory(args.get("path", "."))
        if tool_name == "read_file":
            return self._fs_tool.read_file(args["path"], max_chars=int(args.get("max_chars", 12_000)))
        if tool_name == "write_file":
            return self._fs_tool.write_file(args["path"], args["content"])
        if tool_name == "search_text":
            return self._fs_tool.search_text(
                args["pattern"],
                path=args.get("path", "."),
                max_matches=int(args.get("max_matches", 25)),
            )
        if tool_name == "run_command":
            return self._shell_tool.run(args["command"])
        if tool_name == "web_search":
            return self._web_search_tool.search(args["query"], max_results=int(args.get("max_results", 5)))
        return {"ok": False, "error": f"Unknown tool '{tool_name}'"}

    def update_profile(self, payload: dict[str, str | None]) -> None:
        for key, value in payload.items():
            if not value:
                continue
            self._memory_store.store_memory(
                category="profile",
                title=key,
                content=value,
                tags=["person", "profile", key],
                source="profile_form",
                confidence=0.95,
            )

    def _remember_facts(self, facts: dict[str, str], modality: str) -> None:
        for key, value in facts.items():
            self._memory_store.store_memory(
                category="profile",
                title=key,
                content=value,
                tags=["person", key, modality],
                source=modality,
                confidence=0.92,
            )

    def _store_experience(self, message: str, modality: str, intent_name: str) -> None:
        if intent_name in {"tool_use", "evolve"}:
            return
        if len(message.strip()) < 12:
            return
        self._memory_store.store_memory(
            category="experience",
            title=f"{modality} interaction",
            content=message[:500],
            tags=["experience", modality, intent_name],
            source=modality,
            confidence=0.58,
        )

    def _handle_remember(self, message: str, intent: IntentResult, modality: str) -> str:
        if intent.extracted_facts:
            stored = ", ".join(f"{key}={value}" for key, value in intent.extracted_facts.items())
            return f"Stored durable profile memory from {modality}: {stored}."

        self._memory_store.store_memory(
            category="note",
            title=f"{modality} note",
            content=message,
            tags=["note", modality],
            source=modality,
            confidence=0.74,
        )
        return f"Stored that {modality} note in long-term memory."

    def _handle_memory_query(self) -> str:
        memories = self._memory_store.recall(limit=10)
        if not memories:
            return "I do not have durable memory yet. Speak, type, or show patterns over time and I will accumulate them."

        lines = [f"- [{memory['category']}] {memory['title']}: {memory['content']}" for memory in memories[:8]]
        return "What I currently remember:\n" + "\n".join(lines)

    def _handle_web_search(self, message: str) -> tuple[str, list[dict[str, Any]]]:
        query = re.sub(r"^(search the web for|look up|find online|google)\s+", "", message, flags=re.I).strip()
        result = self._web_search_tool.search(query or message, max_results=5)
        trace = [{"tool": "web_search", "query": query or message, "result": result}]
        if not result.get("ok"):
            return f"Web search failed: {result.get('error', 'unknown error')}", trace
        results = result.get("results", [])
        if not results:
            return "Web search returned no results.", trace
        rendered = "\n".join(
            f"- {item['title']} | {item['href']}\n  {item['body']}" for item in results[:5]
        )
        return f"Top web search results for '{query or message}':\n{rendered}", trace

    def _handle_tool_use(self, message: str) -> tuple[str, list[dict[str, Any]]]:
        lowered = message.lower()
        if "run command" in lowered:
            command = re.sub(r"^.*run command\s+", "", message, flags=re.I)
            result = self.invoke_tool("run_command", {"command": command})
            return self._format_tool_result("run_command", result), [{"tool": "run_command", "result": result}]

        if "read file" in lowered or "open file" in lowered:
            path = self._extract_path(message)
            if not path:
                return "Specify a file path to read.", []
            result = self.invoke_tool("read_file", {"path": path})
            return self._format_tool_result("read_file", result), [{"tool": "read_file", "result": result}]

        if "list files" in lowered or "show directory" in lowered:
            path = self._extract_path(message) or "."
            result = self.invoke_tool("list_directory", {"path": path})
            return self._format_tool_result("list_directory", result), [{"tool": "list_directory", "result": result}]

        if "search text" in lowered or "grep" in lowered:
            pattern = re.sub(r"^.*(?:search text for|grep)\s+", "", message, flags=re.I).strip()
            result = self.invoke_tool("search_text", {"pattern": pattern})
            return self._format_tool_result("search_text", result), [{"tool": "search_text", "result": result}]

        return (
            "Tool use supported: list files, read file <path>, search text for <text>, run command <cmd>, web search <query>.",
            [],
        )

    async def _handle_creation_request(
        self, message: str, recalled_memories: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        if self._message_mentions_repo(message):
            return await self._handle_repo_creation_request(message, recalled_memories)

        if self._llm_adapter.enabled:
            memory_summary = "\n".join(
                f"- {memory['title']}: {memory['content']}" for memory in recalled_memories[:5]
            )
            reply = await self._llm_adapter.complete_text(
                system_prompt=(
                    "You are a concise general-purpose evolving assistant. Help plan or create the requested thing. "
                    "Use user memories when helpful."
                ),
                user_prompt=f"Request: {message}\nRelevant memory:\n{memory_summary or '- none'}",
            )
            if reply:
                return reply, []

        return self._build_rule_based_creation_reply(message, recalled_memories), []

    async def _handle_repo_creation_request(
        self, message: str, recalled_memories: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        keywords = self._extract_keywords(message)
        pattern = " ".join(keywords[:3]) if keywords else message
        search_result = self._fs_tool.search_text(pattern, ".", max_matches=12)
        trace = [{"tool": "search_text", "pattern": pattern, "result": search_result}]

        if self._llm_adapter.enabled:
            prompt = self._build_repo_prompt(message, recalled_memories, search_result)
            reply = await self._llm_adapter.complete_text(
                system_prompt=(
                    "You are a terse repo copilot. Use the provided repository matches and memories. "
                    "Do not invent files or code that were not supplied."
                ),
                user_prompt=prompt,
            )
            if reply:
                return reply, trace

        matches = search_result.get("matches", []) if search_result.get("ok") else []
        if not matches:
            return (
                "I could not find strong repo matches yet. Try a more concrete symbol, file path, or error message.",
                trace,
            )

        rendered = "\n".join(
            f"- {item['path']}:{item['line_number']} -> {item['line']}" for item in matches[:8]
        )
        return (
            "I found candidate local project locations for that request:\n"
            f"{rendered}",
            trace,
        )

    def _handle_evolution(self, message: str) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        if "format" in message.lower():
            result = self._self_improvement.format_repo()
            return result["summary"], [{"tool": "run_command", "result": result["result"]}], []

        result = self._self_improvement.scan()
        return result["summary"], result["tool_trace"], result["insights"]

    async def _handle_conversation(
        self, message: str, recalled_memories: list[dict[str, Any]], modality: str
    ) -> str:
        if self._llm_adapter.enabled:
            memory_summary = "\n".join(
                f"- {memory['title']}: {memory['content']}" for memory in recalled_memories[:5]
            )
            reply = await self._llm_adapter.complete_text(
                system_prompt=(
                    "You are a local-first multimodal agent. You learn from text, voice, and camera over time. "
                    "Be concise and grounded."
                ),
                user_prompt=(
                    f"Modality: {modality}\n"
                    f"User message: {message}\n"
                    f"Relevant memory:\n{memory_summary or '- none'}"
                ),
            )
            if reply:
                return reply

        return (
            "I logged that interaction. I can learn durable facts, keep experiences, observe presence through the camera, accept voice transcripts, search the web, and use local tools when asked."
        )

    def _learn_skill(self, intent_name: str, tool_trace: list[dict[str, Any]], message: str) -> None:
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

    def _message_mentions_repo(self, message: str) -> bool:
        lowered = message.lower()
        return any(
            token in lowered
            for token in ["code", "repo", "repository", "file", "bug", "function", "class", "project", "app"]
        )

    def _extract_path(self, message: str) -> str | None:
        quoted = re.search(r"[`'\"]([^`'\"]+\.[A-Za-z0-9]+)[`'\"]", message)
        if quoted:
            return quoted.group(1)
        path_like = re.search(r"([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)", message)
        if path_like:
            return path_like.group(1)
        directory_like = re.search(r"(?:in|directory)\s+([A-Za-z0-9_./-]+)", message, flags=re.I)
        if directory_like:
            return directory_like.group(1)
        return None

    def _extract_keywords(self, message: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_/-]{2,}", message.lower())
        stopwords = {"help", "with", "this", "that", "about", "please", "build", "write", "make", "create"}
        return [token for token in tokens if token not in stopwords]

    def _build_repo_prompt(
        self, message: str, recalled_memories: list[dict[str, Any]], search_result: dict[str, Any]
    ) -> str:
        memories = "\n".join(f"- {memory['title']}: {memory['content']}" for memory in recalled_memories[:5])
        matches = "\n".join(
            f"- {item['path']}:{item['line_number']} -> {item['line']}"
            for item in search_result.get("matches", [])[:12]
        )
        return (
            f"Task: {message}\n"
            f"Relevant memories:\n{memories or '- none'}\n"
            f"Repository matches:\n{matches or '- none'}\n"
            "Respond with actionable guidance and reference the concrete matches."
        )

    def _build_rule_based_creation_reply(
        self, message: str, recalled_memories: list[dict[str, Any]]
    ) -> str:
        lowered = message.lower()
        memory_hint = ""
        for memory in recalled_memories:
            if memory["category"] == "profile" and memory["title"] in {"habit", "preference", "goal"}:
                memory_hint = f"I’m factoring in your {memory['title']}: {memory['content']}.\n"
                break

        subject = re.sub(r"^(help me|please|can you)\s+", "", message, flags=re.I).strip()
        if "routine" in lowered:
            return (
                f"{memory_hint}Draft routine for: {subject}\n"
                "1. Define the feeling or outcome you want from it.\n"
                "2. Keep the first step frictionless and under five minutes.\n"
                "3. Add one anchor behavior, one core action, and one shutdown cue.\n"
                "4. Run it for three days, then tell me what felt easy or resistant so I can adapt it."
            )

        if "plan" in lowered:
            return (
                f"{memory_hint}Working plan for: {subject}\n"
                "1. State the outcome in one sentence.\n"
                "2. List constraints, tools, and available time.\n"
                "3. Break the work into the smallest next three actions.\n"
                "4. After you try step one, I can refine the plan from the result."
            )

        return (
            f"{memory_hint}I can help shape '{subject}' into a concrete plan. "
            "Tell me the desired outcome, constraints, and first deadline, or mention repo/code context if this should use local tools."
        )

    def _format_tool_result(self, tool_name: str, result: dict[str, Any]) -> str:
        if not result.get("ok"):
            return f"{tool_name} failed: {result.get('error', 'unknown error')}"

        if tool_name == "list_directory":
            items = ", ".join(item["name"] for item in result.get("items", [])[:20])
            return f"Directory {result['path']}: {items}"
        if tool_name == "read_file":
            return f"Contents of {result['path']}:\n{result['content']}"
        if tool_name == "search_text":
            matches = result.get("matches", [])
            if not matches:
                return "No matches found."
            lines = "\n".join(
                f"- {item['path']}:{item['line_number']} -> {item['line']}" for item in matches[:10]
            )
            return f"Search matches:\n{lines}"
        if tool_name == "run_command":
            return (
                f"Command completed with code {result['returncode']}.\n"
                f"STDOUT:\n{result.get('stdout', '')}\nSTDERR:\n{result.get('stderr', '')}"
            )
        return str(result)
