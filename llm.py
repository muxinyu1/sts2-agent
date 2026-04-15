import json
import os
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field
from rich.console import Console
from rich.text import Text


class LLM(BaseModel):

    base_url: str = Field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://llmapi.paratera.com/v1"))
    key: str = Field(default_factory=lambda: os.getenv("LLM_KEY", ""))
    model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "DeepSeek-R1-0528"))

    def _chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def _extract_delta_text(self, delta: dict, key: str) -> str:
        value = delta.get(key)
        if isinstance(value, str):
            return value

        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)

        if isinstance(value, dict):
            text = value.get("text")
            if isinstance(text, str):
                return text

        return ""

    def _stream_split_by_think(
        self,
        chunk: str,
        pending: str,
        mode: Literal["normal", "think"],
    ) -> tuple[str, str, str, Literal["normal", "think"]]:
        start_tag = "<think>"
        end_tag = "</think>"
        pending += chunk
        think_delta_parts: list[str] = []
        normal_delta_parts: list[str] = []

        while pending:
            if mode == "normal":
                start_idx = pending.find(start_tag)
                if start_idx == -1:
                    safe_len = max(0, len(pending) - len(start_tag) + 1)
                    if safe_len == 0:
                        break
                    normal_delta_parts.append(pending[:safe_len])
                    pending = pending[safe_len:]
                    break

                if start_idx > 0:
                    normal_delta_parts.append(pending[:start_idx])
                pending = pending[start_idx + len(start_tag):]
                mode = "think"
                continue

            end_idx = pending.find(end_tag)
            if end_idx == -1:
                safe_len = max(0, len(pending) - len(end_tag) + 1)
                if safe_len == 0:
                    break
                think_delta_parts.append(pending[:safe_len])
                pending = pending[safe_len:]
                break

            if end_idx > 0:
                think_delta_parts.append(pending[:end_idx])
            pending = pending[end_idx + len(end_tag):]
            mode = "normal"

        return (
            "".join(think_delta_parts),
            "".join(normal_delta_parts),
            pending,
            mode,
        )
    
    def make_response(self, system_prompt: str, prompt: str) -> str:
        """
        会抛出异常
        """
        if not self.key:
            raise ValueError("LLM key is empty. Please set environment variable LLM_KEY.")

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
        }

        request = Request(
            self._chat_completions_url(),
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "text/event-stream",
            },
            method="POST",
        )

        console = Console()
        console.print("[bold cyan]LLM Streaming Started[/bold cyan]")

        full_text_parts: list[str] = []
        reasoning_parts: list[str] = []
        pending = ""
        mode: Literal["normal", "think"] = "normal"
        think_header_printed = False
        response_header_printed = False

        try:
            with urlopen(request, timeout=300) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue

                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break

                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = payload.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    if not isinstance(delta, dict):
                        continue

                    reasoning_chunk = (
                        self._extract_delta_text(delta, "reasoning_content")
                        or self._extract_delta_text(delta, "reasoning")
                        or self._extract_delta_text(delta, "thinking")
                    )

                    if reasoning_chunk:
                        reasoning_parts.append(reasoning_chunk)
                        if not think_header_printed:
                            console.print("\n[bold magenta]🧠 THINK[/bold magenta]")
                            think_header_printed = True
                        console.print(Text(reasoning_chunk, style="dim magenta"), end="")

                    chunk = self._extract_delta_text(delta, "content")
                    if not chunk:
                        continue

                    full_text_parts.append(chunk)
                    think_delta, normal_delta, pending, mode = self._stream_split_by_think(
                        chunk=chunk,
                        pending=pending,
                        mode=mode,
                    )

                    if think_delta:
                        if not think_header_printed:
                            console.print("\n[bold magenta]🧠 THINK[/bold magenta]")
                            think_header_printed = True
                        console.print(Text(think_delta, style="dim magenta"), end="")

                    if normal_delta:
                        if not response_header_printed:
                            console.print("\n[bold green]🎯 RESPONSE[/bold green]")
                            response_header_printed = True
                        console.print(Text(normal_delta, style="bold white"), end="")

        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

        if pending:
            if mode == "think":
                if not think_header_printed:
                    console.print("\n[bold magenta]🧠 THINK[/bold magenta]")
                    think_header_printed = True
                console.print(Text(pending, style="dim magenta"), end="")
            else:
                if not response_header_printed:
                    console.print("\n[bold green]🎯 RESPONSE[/bold green]")
                    response_header_printed = True
                console.print(Text(pending, style="bold white"), end="")

        console.print("\n[bold cyan]LLM Streaming Finished[/bold cyan]")

        content_text = "".join(full_text_parts)
        reasoning_text = "".join(reasoning_parts)

        if reasoning_text and "<think>" not in content_text:
            return f"<think>{reasoning_text}</think>\n{content_text}"
        return content_text