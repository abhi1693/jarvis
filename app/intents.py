from __future__ import annotations

import re

from app.llm import LLMAdapter
from app.models import IntentResult, MemoryCandidate


class IntentService:
    _FACT_TITLES = {"name", "role", "current_focus", "goal", "preference", "habit", "feeling"}

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
The agent has no fixed role and should derive duties, constraints, and collaboration style from the operator.
Return JSON with:
- name: one of [conversation, remember, memory_query, create, tool_use, web_search, evolve, orient]
- confidence: 0..1
- extracted_facts: object of durable user facts when obvious
- suggested_tools: array of tool names
- memory_candidates: array of objects with category, title, content, tags, confidence
Use generic categories such as charter, constraint, interaction_style, relationship, objective, person, preference, note.
Only return JSON.
""".strip()

        try:
            response = await self._llm_adapter.complete_json(system_prompt, f"Message: {message}")
        except Exception:
            return None

        if not response or "name" not in response:
            return None

        memory_candidates = self._coerce_memory_candidates(response.get("memory_candidates", []))
        extracted_facts = {
            str(key): str(value)
            for key, value in dict(response.get("extracted_facts", {})).items()
        }
        if not memory_candidates and extracted_facts:
            memory_candidates = [
                self._candidate_from_fact(title, content)
                for title, content in extracted_facts.items()
            ]

        return IntentResult(
            name=str(response.get("name", "conversation")),
            confidence=float(response.get("confidence", 0.0)),
            extracted_facts=extracted_facts,
            suggested_tools=[str(item) for item in response.get("suggested_tools", [])],
            memory_candidates=memory_candidates,
        )

    def _parse_with_rules(self, message: str) -> IntentResult:
        lowered = message.lower()
        memory_candidates = self._extract_memory_candidates(message)
        facts = {
            candidate.title: candidate.content
            for candidate in memory_candidates
            if candidate.title in self._FACT_TITLES
        }
        suggested_tools: list[str] = []
        intent_name = "conversation"
        confidence = 0.35

        if self._looks_like_orientation(lowered, memory_candidates):
            intent_name = "orient"
            confidence = 0.88
        elif self._contains_any(lowered, ["remember", "store this", "note that", "save this"]):
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
            memory_candidates=memory_candidates,
        )

    def _extract_memory_candidates(self, message: str) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        seen: set[tuple[str, str, str]] = set()

        def add_candidate(
            category: str,
            title: str,
            content: str,
            tags: list[str],
            confidence: float,
        ) -> None:
            normalized = content.strip(" .,!?\n\t")
            if not normalized:
                return
            key = (category, title, normalized.lower())
            if key in seen:
                return
            seen.add(key)
            candidates.append(
                MemoryCandidate(
                    category=category,
                    title=title,
                    content=normalized,
                    tags=tags,
                    confidence=confidence,
                )
            )

        pattern_map = [
            ("person", "name", [r"\bmy name is ([A-Za-z][A-Za-z0-9 _-]{1,50}?)(?:\s+(?:and|but)\b|[.!?,]|$)", r"\bcall me ([A-Za-z][A-Za-z0-9 _-]{1,50}?)(?:\s+(?:and|but)\b|[.!?,]|$)"], ["person", "identity"], 0.94),
            ("person", "role", [r"\bi am a[n]? ([A-Za-z][A-Za-z0-9 _-]{1,60})"], ["person", "identity"], 0.88),
            ("objective", "current_focus", [r"\bi(?:'m| am) working on ([^.!\n]+)"], ["objective", "current"], 0.82),
            ("objective", "goal", [r"\bmy goal is ([^.!\n]+)"], ["objective", "goal"], 0.86),
            ("preference", "preference", [r"\bi (?:prefer|like|love) ([^.!\n]+)"], ["preference", "user_model"], 0.8),
            ("preference", "habit", [r"\bi usually ([^.!\n]+)"], ["habit", "user_model"], 0.72),
            ("state", "feeling", [r"\bi feel ([^.!\n]+)"], ["state"], 0.68),
            ("relationship", "operator_role", [r"\bi am your ([^.!\n]+)"], ["relationship", "operator"], 0.86),
            ("charter", "duty", [r"\b(?:your|the app(?:'s)?|the system(?:'s)?|this app(?:'s)?|this system(?:'s)?) (?:job|role|purpose|duty) is to ([^.!\n]+)"], ["charter", "duty"], 0.94),
            ("charter", "duty", [r"\bi want you to ([^.!\n]+)", r"\byou are here to ([^.!\n]+)", r"\bthis (?:app|system) should ([^.!\n]+)", r"\bthis (?:app|system) must ([^.!\n]+)"], ["charter", "duty"], 0.88),
            ("constraint", "avoid", [r"\b(?:you|this (?:app|system)) (?:must not|should not|cannot|can't) ([^.!\n]+)", r"\bdo not ([^.!\n]+)", r"\bavoid ([^.!\n]+)"], ["charter", "constraint"], 0.84),
            ("interaction_style", "style", [r"\b(?:respond|speak|interact|work with me) (?:in|with|using) ([^.!\n]+)"], ["style", "interaction"], 0.74),
            ("objective", "learning_focus", [r"\blearn about me(?: by| through| from)? ([^.!\n]+)"], ["learning", "objective"], 0.78),
        ]

        for category, title, expressions, tags, confidence in pattern_map:
            for expression in expressions:
                match = re.search(expression, message, flags=re.IGNORECASE)
                if match:
                    add_candidate(category, title, match.group(1), tags, confidence)
                    break

        return candidates

    def _looks_like_orientation(self, lowered: str, candidates: list[MemoryCandidate]) -> bool:
        if any(candidate.category in {"charter", "constraint", "interaction_style", "relationship"} for candidate in candidates):
            return True

        return self._contains_any(
            lowered,
            [
                "your job is",
                "your role is",
                "your purpose is",
                "i want you to",
                "you are here to",
                "this app should",
                "this system should",
                "this app must",
                "this system must",
                "adapt to me",
                "learn about me",
            ],
        )

    def _coerce_memory_candidates(self, raw_candidates: object) -> list[MemoryCandidate]:
        if not isinstance(raw_candidates, list):
            return []

        candidates: list[MemoryCandidate] = []
        for item in raw_candidates:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            candidates.append(
                MemoryCandidate(
                    category=str(item.get("category", "note")),
                    title=str(item.get("title", "note")),
                    content=content,
                    tags=[str(tag) for tag in item.get("tags", [])],
                    confidence=float(item.get("confidence", 0.75)),
                )
            )
        return candidates

    def _candidate_from_fact(self, title: str, content: str) -> MemoryCandidate:
        category_map = {
            "name": "person",
            "role": "person",
            "current_focus": "objective",
            "goal": "objective",
            "preference": "preference",
            "habit": "preference",
            "feeling": "state",
        }
        category = category_map.get(title, "note")
        tags = [category, title]
        return MemoryCandidate(
            category=category,
            title=title,
            content=content,
            tags=tags,
            confidence=0.78,
        )

    def _contains_any(self, value: str, options: list[str]) -> bool:
        return any(option in value for option in options)
