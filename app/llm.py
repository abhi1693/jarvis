from __future__ import annotations

import asyncio
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import httpx

from app.config import Settings


class _CodexUnsupportedModelError(RuntimeError):
    """Raised when a requested Codex model is unavailable for the current account."""


class LLMAdapter:
    def __init__(self, settings: Settings):
        self._settings = settings

    @property
    def enabled(self) -> bool:
        return self._settings.llm_enabled

    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        if self._settings.llm_http_enabled:
            message = await self._complete_http_chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0,
                timeout_seconds=self._settings.llm_timeout_seconds,
            )
            parsed = self._extract_json(message) if message else None
            if parsed is not None:
                return parsed

        if not self._settings.codex_cli_enabled:
            return None

        message = await self._run_codex_cli(
            prompt=self._build_codex_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                require_json=True,
            )
        )
        return self._extract_json(message) if message else None

    async def complete_text(self, system_prompt: str, user_prompt: str) -> str | None:
        if self._settings.llm_http_enabled:
            message = await self._complete_http_chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2,
                timeout_seconds=self._settings.llm_timeout_seconds,
            )
            if message:
                return message

        if not self._settings.codex_cli_enabled:
            return None

        return await self._run_codex_cli(
            prompt=self._build_codex_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                require_json=False,
            )
        )

    async def _complete_http_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        timeout_seconds: int,
    ) -> str | None:
        headers = {"Content-Type": "application/json"}
        if self._settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self._settings.llm_api_key}"

        payload = {
            "model": self._settings.llm_model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(
                    f"{self._settings.llm_compat_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
            message = str(data["choices"][0]["message"]["content"]).strip()
        except (httpx.HTTPError, IndexError, KeyError, TypeError, ValueError):
            return None

        return message or None

    async def _run_codex_cli(
        self,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
    ) -> str | None:
        model = self._settings.codex_cli_model
        try:
            return await self._run_codex_cli_once(
                prompt=prompt,
                output_schema=output_schema,
                model=model,
            )
        except _CodexUnsupportedModelError:
            if not model:
                return None
            try:
                return await self._run_codex_cli_once(
                    prompt=prompt,
                    output_schema=output_schema,
                    model=None,
                )
            except (OSError, TimeoutError, _CodexUnsupportedModelError):
                return None
        except (OSError, TimeoutError):
            return None

    async def _run_codex_cli_once(
        self,
        prompt: str,
        output_schema: dict[str, Any] | None,
        model: str | None,
    ) -> str | None:
        if not self._settings.codex_cli_path:
            return None

        with tempfile.TemporaryDirectory(prefix="jarvis-codex-") as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / "response.txt"
            command = [
                self._settings.codex_cli_path,
                "exec",
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--ephemeral",
                "-C",
                str(self._settings.repo_root),
                "-o",
                str(output_path),
            ]
            if model:
                command.extend(["-m", model])
            if output_schema is not None:
                schema_path = temp_path / "schema.json"
                schema_path.write_text(json.dumps(output_schema), encoding="utf-8")
                command.extend(["--output-schema", str(schema_path)])
            command.append(prompt)

            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self._settings.llm_timeout_seconds,
                )
            except TimeoutError:
                process.kill()
                await process.communicate()
                raise

            stderr_text = stderr.decode("utf-8", errors="replace")
            if process.returncode != 0:
                if model and self._is_unsupported_codex_model_error(stderr_text):
                    raise _CodexUnsupportedModelError(stderr_text.strip())
                return None

            if output_path.exists():
                message = output_path.read_text(encoding="utf-8").strip()
                if message:
                    return message

            message = stdout.decode("utf-8", errors="replace").strip()
            return message or None

    def _build_codex_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        require_json: bool,
    ) -> str:
        response_instruction = (
            "Return only a valid JSON object."
            if require_json
            else "Respond directly with the answer only."
        )
        return (
            "You are acting only as the language-model backend for a local agent.\n"
            "Do not inspect the repository, run shell commands, or use tools.\n"
            "Use only the instructions below.\n"
            f"{response_instruction}\n\n"
            f"System prompt:\n{system_prompt}\n\n"
            f"User prompt:\n{user_prompt}"
        )

    def _is_unsupported_codex_model_error(self, stderr_text: str) -> bool:
        lowered = stderr_text.lower()
        return "model is not supported" in lowered and "chatgpt account" in lowered

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
