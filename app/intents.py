from __future__ import annotations

import re

from app.llm import LLMAdapter
from app.models import IntentResult


class IntentService:
    def __init__(self, llm_adapter: LLMAdapter):
        self._llm_adapter = llm_adapter

    async def parse(self, message: str) -> IntentResult:
        llm_result = await self._parse_with_llm(message)
        if llm_result is not None:
            return llm_result
        return self._parse_with_rules(message)

    async def _parse_with_llm(self, message: str) -> IntentResult | None:
        if not self._llm_adapter.enabled:
            return None

        system_prompt = """
You classify user intent for a local multimodal agent that learns from text, voice, and camera input.
Return JSON with:
- name: one of [conversation, remember, memory_query, create, tool_use, web_search, evolve]
- confidence: 0..1
- extracted_facts: object of durable user facts or preferences
- suggested_tools: array of tool names
Only return JSON.
""".strip()

        try:
            response = await self._llm_adapter.complete_json(system_prompt, f"Message: {message}")
        except Exception:
            return None

        if not response or "name" not in response:
            return None

        return IntentResult(
            name=str(response.get("name", "conversation")),
            confidence=float(response.get("confidence", 0.0)),
            extracted_facts={
                str(key): str(value)
                for key, value in dict(response.get("extracted_facts", {})).items()
            },
            suggested_tools=[str(item) for item in response.get("suggested_tools", [])],
        )

    def _parse_with_rules(self, message: str) -> IntentResult:
        lowered = message.lower()
        facts = self._extract_facts(message)
        suggested_tools: list[str] = []
        intent_name = "conversation"
        confidence = 0.35

        if self._contains_any(lowered, ["remember", "store this", "note that", "save this"]):
            intent_name = "remember"
            confidence = 0.8
        elif self._contains_any(lowered, ["what do you know about me", "what do you remember", "what have you learned"]):
            intent_name = "memory_query"
            confidence = 0.86
        elif self._contains_any(lowered, ["search the web", "look up", "find online", "google"]):
            intent_name = "web_search"
            confidence = 0.82
            suggested_tools.append("web_search")
        elif self._contains_any(
            lowered,
            ["evolve", "reflect", "scan yourself", "improve yourself", "audit yourself", "review yourself"],
        ):
            intent_name = "evolve"
            confidence = 0.88
            suggested_tools.extend(["search_text", "run_command"])
        elif self._contains_any(
            lowered,
            ["list files", "read file", "open file", "search text", "grep", "run command", "show directory"],
        ):
            intent_name = "tool_use"
            confidence = 0.84
            suggested_tools.extend(["list_directory", "read_file", "search_text", "run_command"])
        elif facts and not self._contains_any(lowered, ["help me", "can you", "please", "make me", "build me"]):
            intent_name = "remember"
            confidence = 0.72
        elif self._contains_any(
            lowered,
            ["build", "write", "plan", "make", "design", "fix", "help me", "create", "implement", "debug"],
        ):
            intent_name = "create"
            confidence = 0.76
            if self._contains_any(lowered, ["code", "repo", "file", "bug", "function", "app"]):
                suggested_tools.extend(["search_text", "read_file"])

        return IntentResult(
            name=intent_name,
            confidence=confidence,
            extracted_facts=facts,
            suggested_tools=suggested_tools,
        )

    def _extract_facts(self, message: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        patterns = {
            "name": [
                r"\bmy name is ([A-Za-z][A-Za-z0-9 _-]{1,50}?)(?:\s+(?:and|but)\b|[.!?,]|$)",
                r"\bcall me ([A-Za-z][A-Za-z0-9 _-]{1,50}?)(?:\s+(?:and|but)\b|[.!?,]|$)",
            ],
            "role": [r"\bi am a[n]? ([A-Za-z][A-Za-z0-9 _-]{1,60})"],
            "current_focus": [r"\bi(?:'m| am) working on ([^.!\n]+)"],
            "goal": [r"\bmy goal is ([^.!\n]+)"],
            "preference": [r"\bi (?:prefer|like|love) ([^.!\n]+)"],
            "habit": [r"\bi usually ([^.!\n]+)"],
            "feeling": [r"\bi feel ([^.!\n]+)"],
        }

        for key, expressions in patterns.items():
            for expression in expressions:
                match = re.search(expression, message, flags=re.IGNORECASE)
                if match:
                    facts[key] = match.group(1).strip()
                    break

        return facts

    def _contains_any(self, value: str, options: list[str]) -> bool:
        return any(option in value for option in options)
