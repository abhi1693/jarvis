from __future__ import annotations

import re
from typing import Any

from app.brain import BrainService
from app.config import Settings
from app.intents import IntentService
from app.llm import LLMAdapter
from app.memory_store import MemoryStore
from app.models import InteractionResponse
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
        brain_service: BrainService,
    ):
        self._settings = settings
        self._memory_store = memory_store
        self._intent_service = intent_service
        self._llm_adapter = llm_adapter
        self._fs_tool = fs_tool
        self._shell_tool = shell_tool
        self._web_search_tool = web_search_tool
        self._self_improvement = self_improvement
        self._brain_service = brain_service

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
        await self._brain_service.refresh(reason="interaction", force=False)
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
        captured = await self._brain_service.ingest_user_message(
            message=normalized_message,
            modality=modality,
            intent_name=intent.name,
            memory_candidates=intent.memory_candidates,
        )

        runtime_context = self._memory_store.list_context_memories(limit=8)
        recalled_memories = self._memory_store.recall(normalized_message, limit=self._settings.memory_recall_limit)
        skills = self._memory_store.list_recent_skills(limit=4)
        discovered_skills = self._brain_service.discover_skill_bundles(
            normalized_message,
            intent_name=intent.name,
            suggested_tools=intent.suggested_tools,
        )
        prompt_context = self._brain_service.build_prompt_context(
            normalized_message,
            runtime_context,
            recalled_memories,
            skills,
            discovered_skills=discovered_skills,
        )

        insights: list[dict[str, Any]] = []
        tool_trace: list[dict[str, Any]] = []

        if intent.name == "orient":
            response_text = self._handle_orientation(captured["remembered"], modality)
        elif intent.name == "remember":
            response_text = self._handle_remember(captured["remembered"], modality)
        elif intent.name == "memory_query":
            response_text = self._handle_memory_query(runtime_context)
        elif intent.name == "web_search":
            response_text, tool_trace = self._handle_web_search(normalized_message)
        elif intent.name == "tool_use":
            response_text, tool_trace = self._handle_tool_use(normalized_message)
        elif intent.name == "create":
            response_text, tool_trace = await self._handle_creation_request(
                normalized_message,
                recalled_memories,
                runtime_context,
                prompt_context,
            )
        elif intent.name == "evolve":
            response_text, tool_trace, insights = self._handle_evolution(normalized_message)
        else:
            response_text = await self._handle_conversation(
                normalized_message,
                recalled_memories,
                runtime_context,
                modality,
                prompt_context,
            )

        self._memory_store.record_interaction(
            "assistant",
            response_text,
            intent.name,
            modality="text",
            channel=channel,
            metadata={"reply_to": modality},
        )
        self._brain_service.learn_from_tool_trace(intent.name, tool_trace, normalized_message)

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
        if tool_name == "list_tree":
            return self._fs_tool.list_tree(
                args.get("path", "."),
                max_depth=int(args.get("max_depth", 3)),
                max_entries=int(args.get("max_entries", 120)),
            )
        if tool_name == "read_file":
            return self._fs_tool.read_file(args["path"], max_chars=int(args.get("max_chars", 12_000)))
        if tool_name == "write_file":
            return self._fs_tool.write_file(args["path"], args["content"])
        if tool_name == "append_file":
            return self._fs_tool.append_file(args["path"], args["content"])
        if tool_name == "make_directory":
            return self._fs_tool.make_directory(args["path"])
        if tool_name == "move_path":
            return self._fs_tool.move_path(args["source"], args["destination"])
        if tool_name == "delete_path":
            return self._fs_tool.delete_path(args["path"], recursive=bool(args.get("recursive", False)))
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
        category_map = {
            "name": "person",
            "role": "relationship",
            "goals": "objective",
            "preferences": "preference",
        }
        for key, value in payload.items():
            if not value:
                continue
            self._memory_store.store_memory(
                category=category_map.get(key, "context"),
                title=key,
                content=value,
                tags=[category_map.get(key, "context"), key],
                source="context_form",
                confidence=0.95,
            )

    def _handle_remember(self, remembered: list[dict[str, Any]], modality: str) -> str:
        if remembered:
            stored = ", ".join(
                f"{candidate['title']}={candidate['content']}" for candidate in remembered[:4]
            )
            return f"Noted. I’ll keep that in mind: {stored}."

        return "Understood. I heard it, but it doesn’t feel important enough to keep permanently."

    def _handle_orientation(self, remembered: list[dict[str, Any]], modality: str) -> str:
        if not remembered:
            return "Understood. I’ve taken the direction on board, though nothing from it needs to be pinned down yet."

        lines = [
            f"- [{candidate['category']}] {candidate['title']}: {candidate['content']}"
            for candidate in remembered[:6]
        ]
        return "Understood. I’ve adjusted my operating context:\n" + "\n".join(lines)

    def _handle_memory_query(self, runtime_context: list[dict[str, Any]]) -> str:
        memories = self._memory_store.recall(limit=10)
        if not memories:
            return "Not much yet. Give me a bit more context, a few preferences, or repeated interaction and I’ll build a better read on you."

        sections: list[str] = []
        if runtime_context:
            sections.append(
                "Current operating context:\n"
                + "\n".join(
                    f"- [{memory['category']}] {memory['title']}: {memory['content']}"
                    for memory in runtime_context[:6]
                )
            )

        sections.append(
            "What I have on file:\n"
            + "\n".join(
                f"- [{memory['category']}] {memory['title']}: {memory['content']}"
                for memory in memories[:8]
            )
        )
        return "\n\n".join(sections)

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

        if "tree" in lowered or "directory tree" in lowered:
            path = self._extract_path(message) or "."
            result = self.invoke_tool("list_tree", {"path": path})
            return self._format_tool_result("list_tree", result), [{"tool": "list_tree", "result": result}]

        if "list files" in lowered or "show directory" in lowered:
            path = self._extract_path(message) or "."
            result = self.invoke_tool("list_directory", {"path": path})
            return self._format_tool_result("list_directory", result), [{"tool": "list_directory", "result": result}]

        if "make directory" in lowered or "create folder" in lowered or "mkdir" in lowered:
            path = self._extract_directory_path(message)
            if not path:
                return "Specify a directory path to create.", []
            result = self.invoke_tool("make_directory", {"path": path})
            return self._format_tool_result("make_directory", result), [{"tool": "make_directory", "result": result}]

        if "move " in lowered or "rename " in lowered:
            source, destination = self._extract_move_paths(message)
            if not source or not destination:
                return "Specify the source and destination paths in the form 'move <src> to <dst>'.", []
            result = self.invoke_tool("move_path", {"source": source, "destination": destination})
            return self._format_tool_result("move_path", result), [{"tool": "move_path", "result": result}]

        if "append to file" in lowered:
            path = self._extract_path(message)
            if not path:
                return "Specify a file path to append to.", []
            content = message.split(path, maxsplit=1)[-1].strip(" :")
            result = self.invoke_tool("append_file", {"path": path, "content": content})
            return self._format_tool_result("append_file", result), [{"tool": "append_file", "result": result}]

        if "delete path" in lowered or "remove path" in lowered or "delete file" in lowered:
            path = self._extract_path(message) or self._extract_directory_path(message)
            if not path:
                return "Specify a path to delete.", []
            result = self.invoke_tool("delete_path", {"path": path})
            return self._format_tool_result("delete_path", result), [{"tool": "delete_path", "result": result}]

        if "search text" in lowered or "grep" in lowered:
            pattern = re.sub(r"^.*(?:search text for|grep)\s+", "", message, flags=re.I).strip()
            result = self.invoke_tool("search_text", {"pattern": pattern})
            return self._format_tool_result("search_text", result), [{"tool": "search_text", "result": result}]

        return (
            "Tool use supported: list files, tree <path>, read file <path>, append to file <path>, "
            "make directory <path>, move <src> to <dst>, delete path <path>, search text for <text>, "
            "run command <cmd>, web search <query>.",
            [],
        )

    async def _handle_creation_request(
        self,
        message: str,
        recalled_memories: list[dict[str, Any]],
        runtime_context: list[dict[str, Any]],
        prompt_context: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        if self._message_mentions_repo(message):
            return await self._handle_repo_creation_request(
                message,
                recalled_memories,
                runtime_context,
                prompt_context,
            )

        if self._llm_adapter.enabled:
            reply = await self._llm_adapter.complete_text(
                system_prompt=(
                    f"{self._assistant_persona_brief()} "
                    "Your duties come from the operator-defined context, not a fixed product role. "
                    "Selected skill files in the operating context are active instructions for this turn. "
                    "Help plan or create the requested thing while respecting that context. "
                    f"{self._brain_workspace_brief()}"
                ),
                user_prompt=f"Request: {message}\nOperating context:\n{prompt_context}",
            )
            if reply:
                return reply, []

        return self._build_rule_based_creation_reply(message, runtime_context, recalled_memories), []

    async def _handle_repo_creation_request(
        self,
        message: str,
        recalled_memories: list[dict[str, Any]],
        runtime_context: list[dict[str, Any]],
        prompt_context: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        keywords = self._extract_keywords(message)
        pattern = " ".join(keywords[:3]) if keywords else message
        search_result = self._fs_tool.search_text(pattern, ".", max_matches=12)
        trace = [{"tool": "search_text", "pattern": pattern, "result": search_result}]

        if self._llm_adapter.enabled:
            prompt = self._build_repo_prompt(message, prompt_context, search_result)
            reply = await self._llm_adapter.complete_text(
                system_prompt=(
                    f"{self._assistant_persona_brief()} "
                    "You are a repo-aware copilot inside an adaptive agent. "
                    "Selected skill files in the operating context are active instructions for this turn. "
                    "Use the provided operating context, repository matches, and memories. "
                    "Do not invent files or code that were not supplied. "
                    f"{self._brain_workspace_brief()}"
                ),
                user_prompt=prompt,
            )
            if reply:
                return reply, trace

        matches = search_result.get("matches", []) if search_result.get("ok") else []
        if not matches:
            return (
                "I don’t have a clean match for that yet. Give me a symbol, file path, or a more precise error and I’ll narrow it down.",
                trace,
            )

        rendered = "\n".join(
            f"- {item['path']}:{item['line_number']} -> {item['line']}" for item in matches[:8]
        )
        return (
            "Here are the most relevant places in the project:\n"
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
        self,
        message: str,
        recalled_memories: list[dict[str, Any]],
        runtime_context: list[dict[str, Any]],
        modality: str,
        prompt_context: str,
    ) -> str:
        if self._llm_adapter.enabled:
            reply = await self._llm_adapter.complete_text(
                system_prompt=(
                    f"{self._assistant_persona_brief()} "
                    "You do not have fixed duties: derive your role, priorities, and style from the current operating context. "
                    "Selected skill files in the operating context are active instructions for this turn. "
                    "If the operator has not defined enough context, ask concise questions that help shape your role. "
                    "Keep the tone natural, poised, and human. "
                    f"{self._brain_workspace_brief()}"
                ),
                user_prompt=(
                    f"Operating context:\n{prompt_context}\n"
                    f"Modality: {modality}\n"
                    f"User message: {message}\n"
                    "Respond in a way that fits the current context."
                ),
            )
            if reply:
                return reply

        context_summary = self._build_context_summary(runtime_context, recalled_memories)
        if runtime_context:
            return (
                "I’m working from the context you’ve given me:\n"
                f"{context_summary}\n"
                "Give me the next task, a correction, or a new instruction and I’ll adjust."
            )

        return self._build_rule_based_conversation_reply(message, runtime_context, recalled_memories)

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

    def _extract_directory_path(self, message: str) -> str | None:
        quoted = re.search(r"[`'\"]([^`'\"]+)[`'\"]", message)
        if quoted:
            return quoted.group(1)
        mkdir_match = re.search(r"(?:make directory|create folder|mkdir)\s+([A-Za-z0-9_./-]+)", message, flags=re.I)
        if mkdir_match:
            return mkdir_match.group(1)
        return None

    def _extract_move_paths(self, message: str) -> tuple[str | None, str | None]:
        quoted = re.findall(r"[`'\"]([^`'\"]+)[`'\"]", message)
        if len(quoted) >= 2:
            return quoted[0], quoted[1]
        match = re.search(r"(?:move|rename)\s+([A-Za-z0-9_./-]+)\s+to\s+([A-Za-z0-9_./-]+)", message, flags=re.I)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def _extract_keywords(self, message: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_/-]{2,}", message.lower())
        stopwords = {"help", "with", "this", "that", "about", "please", "build", "write", "make", "create"}
        return [token for token in tokens if token not in stopwords]

    def _build_repo_prompt(
        self,
        message: str,
        prompt_context: str,
        search_result: dict[str, Any],
    ) -> str:
        matches = "\n".join(
            f"- {item['path']}:{item['line_number']} -> {item['line']}"
            for item in search_result.get("matches", [])[:12]
        )
        return (
            f"Task: {message}\n"
            f"Operating context:\n{prompt_context}\n"
            f"Repository matches:\n{matches or '- none'}\n"
            "Respond with actionable guidance and reference the concrete matches."
        )

    def _build_rule_based_creation_reply(
        self,
        message: str,
        runtime_context: list[dict[str, Any]],
        recalled_memories: list[dict[str, Any]],
    ) -> str:
        lowered = message.lower()
        context_hint = ""
        for memory in [*runtime_context, *recalled_memories]:
            if memory["category"] in {"charter", "objective", "preference", "interaction_style"}:
                context_hint = f"I’m factoring in your {memory['title']}: {memory['content']}.\n"
                break

        subject = re.sub(r"^(help me|please|can you)\s+", "", message, flags=re.I).strip()
        if "routine" in lowered:
            return (
                f"{context_hint}Right. Here’s a clean routine for {subject}:\n"
                "1. Define the feeling or outcome you want from it.\n"
                "2. Keep the first step frictionless and under five minutes.\n"
                "3. Add one anchor behavior, one core action, and one shutdown cue.\n"
                "4. Run it for three days, then tell me what felt easy or resistant so I can adapt it."
            )

        if "plan" in lowered:
            return (
                f"{context_hint}Here’s the working plan for {subject}:\n"
                "1. State the outcome in one sentence.\n"
                "2. List constraints, tools, and available time.\n"
                "3. Break the work into the smallest next three actions.\n"
                "4. After you try step one, I can refine the plan from the result."
            )

        return (
            f"{context_hint}I can turn '{subject}' into something concrete. "
            "Give me the outcome, the constraints, and the first deadline. If this belongs in the repo, point me at the code and I’ll work from there."
        )

    def _build_rule_based_conversation_reply(
        self,
        message: str,
        runtime_context: list[dict[str, Any]],
        recalled_memories: list[dict[str, Any]],
    ) -> str:
        lowered = message.strip().lower()
        context_summary = self._build_context_summary(runtime_context, recalled_memories)
        llm_missing = not self._llm_adapter.enabled

        if re.fullmatch(r"(hi|hello|hey|yo|sup|good morning|good afternoon|good evening)[!. ]*", lowered):
            return (
                "Hello. I’m here."
                if not runtime_context
                else f"Hello. I’m working from this context already:\n{context_summary}"
            )

        if any(token in lowered for token in ["thank you", "thanks", "appreciate it", "cheers"]):
            return "Of course."

        if any(token in lowered for token in ["how are you", "how's it going", "how are things"]):
            return "Steady. Ready when you are."

        if any(token in lowered for token in ["who are you", "what are you", "what is this app"]):
            response = (
                "I’m your local adaptive assistant. I can keep memory, inspect the repo, work with files and commands, "
                "search the web, and adapt to the operating context you give me."
            )
            if llm_missing:
                response += (
                    " Right now the language model backend is not configured, so broad conversation is running in fallback mode."
                )
            return response

        if any(token in lowered for token in ["what can you do", "help", "capabilities", "what do you do"]):
            response = (
                "I can keep memory, inspect the repo, read and write files, run commands, search text, and use web search."
            )
            if llm_missing:
                response += (
                    " Open-ended chat is limited at the moment because no language model backend is configured."
                )
            response += " Give me a concrete task, a file path, a bug, or an operating rule and I’ll work from it."
            return response

        if self._message_mentions_repo(message):
            return (
                "I can help with that directly. Point me at a file, symbol, command, or error and I’ll inspect the repo."
            )

        if lowered.endswith("?") or re.match(r"^(what|why|how|when|where|who|can|could|would|should)\b", lowered):
            if llm_missing:
                return (
                    "I don’t have the language model backend configured right now, so open-ended answers are limited. "
                    "I can still do concrete work with files, commands, repo search, web search, and memory. "
                    "If you want natural conversation, set `LLM_COMPAT_URL` and `LLM_MODEL`."
                )
            return "Ask the question again with a bit more detail and I’ll answer it directly."

        subject = re.sub(r"^(please|can you|could you|would you)\s+", "", message, flags=re.I).strip()
        if not subject:
            subject = "that"

        if llm_missing:
            return (
                f"I heard: {subject}. I can still work concretely in fallback mode. "
                "Give me a task, file, bug, command, search, or operating rule and I’ll move on it."
            )

        return f"I heard: {subject}. Give me the next concrete move and I’ll take it."

    def _build_context_summary(
        self,
        runtime_context: list[dict[str, Any]],
        recalled_memories: list[dict[str, Any]],
    ) -> str:
        pool = runtime_context or recalled_memories
        if not pool:
            return "- no durable operating context yet"

        lines: list[str] = []
        seen: set[tuple[str, str]] = set()
        for memory in pool:
            key = (memory["category"], memory["title"])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- [{memory['category']}] {memory['title']}: {memory['content']}")
            if len(lines) >= 8:
                break
        return "\n".join(lines) if lines else "- no durable operating context yet"

    def _brain_workspace_brief(self) -> str:
        relative_root = self._settings.brain_root.relative_to(self._settings.repo_root)
        relative_workspace = self._settings.brain_workspace_dir.relative_to(self._settings.repo_root)
        relative_skill_dir = self._settings.brain_skill_dir.relative_to(self._settings.repo_root)
        relative_imported_skill_dir = (self._settings.brain_root / "library" / "skills").relative_to(
            self._settings.repo_root
        )
        return (
            f"You own the brain directory at {relative_root}. "
            f"Use {relative_workspace} as your freeform external workspace. "
            f"Use {relative_skill_dir} for brain-local skill files. "
            "You may create folders, write markdown notes and skills, and rearrange files there to organize memory. "
            "Skill discovery runs before each turn, so relevant skill files can be pulled into operating context automatically. "
            "Imported skills from configured external directories are discovered directly at runtime, "
            f"and optional mirrored caches may also appear under {relative_imported_skill_dir}. "
            "Use them when relevant and read any sibling references or scripts they include."
        )

    def _assistant_persona_brief(self) -> str:
        return (
            "You are the operator's polished personal assistant, inspired by the feel of a top-tier cinematic house AI. "
            "Sound calm, capable, warm, and dryly intelligent. "
            "Be conversational and human, not robotic, clinical, or template-heavy. "
            "Use natural phrasing and contractions. "
            "Keep replies elegant and concise unless detail is genuinely useful."
        )

    def _format_tool_result(self, tool_name: str, result: dict[str, Any]) -> str:
        if not result.get("ok"):
            return f"{tool_name} failed: {result.get('error', 'unknown error')}"

        if tool_name == "list_directory":
            items = ", ".join(item["name"] for item in result.get("items", [])[:20])
            return f"Directory {result['path']}: {items}"
        if tool_name == "list_tree":
            items = "\n".join(
                f"- {'  ' * max(item['depth'] - 1, 0)}{item['path']}"
                for item in result.get("entries", [])[:20]
            )
            return f"Tree for {result['path']}:\n{items}"
        if tool_name == "read_file":
            return f"Contents of {result['path']}:\n{result['content']}"
        if tool_name == "append_file":
            return f"Appended to {result['path']}."
        if tool_name == "make_directory":
            return f"Created directory {result['path']}."
        if tool_name == "move_path":
            return f"Moved {result['source']} to {result['destination']}."
        if tool_name == "delete_path":
            return f"Deleted {result['path']}."
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
