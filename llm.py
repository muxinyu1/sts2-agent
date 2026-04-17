import json
import os
import time
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field
from rich.console import Console
from rich.status import Status
from rich.text import Text


class LLM(BaseModel):

    base_url: str = "https://llmapi.paratera.com/v1"
    key: str = Field(default_factory=lambda: os.getenv("LLM_KEY", ""))
    model: str = "DeepSeek-R1-0528"
    temperature: float = 0.45
    top_p: float = 0.90
    max_tokens: int = 2048
    max_token_retries: int = 1

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

    def _extract_tag_content(self, text: str, tag: str) -> str:
        if not isinstance(text, str) or not text:
            return ""

        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"

        start = text.find(open_tag)
        if start == -1:
            return ""
        start += len(open_tag)

        end = text.find(close_tag, start)
        if end == -1:
            return ""

        return text[start:end].strip()

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

    def _stream_split_by_tag(
        self,
        chunk: str,
        pending: str,
        mode: Literal["normal", "tag"],
        tag: str,
    ) -> tuple[str, str, str, Literal["normal", "tag"]]:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        pending += chunk
        tag_delta_parts: list[str] = []
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
                mode = "tag"
                continue

            end_idx = pending.find(end_tag)
            if end_idx == -1:
                safe_len = max(0, len(pending) - len(end_tag) + 1)
                if safe_len == 0:
                    break
                tag_delta_parts.append(pending[:safe_len])
                pending = pending[safe_len:]
                break

            if end_idx > 0:
                tag_delta_parts.append(pending[:end_idx])
            pending = pending[end_idx + len(end_tag):]
            mode = "normal"

        return (
            "".join(tag_delta_parts),
            "".join(normal_delta_parts),
            pending,
            mode,
        )
    
    def make_response(
        self,
        system_prompt: str,
        prompt: str,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        """
        会抛出异常
        """
        if not self.key:
            raise ValueError("LLM key is empty. Please set environment variable LLM_KEY.")

        req_temperature = self.temperature if temperature is None else float(temperature)
        req_top_p = self.top_p if top_p is None else float(top_p)
        req_max_tokens = max(1, int(self.max_tokens))
        retry_limit = max(0, int(self.max_token_retries))

        console = Console()

        for attempt in range(retry_limit + 1):
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": req_temperature,
                "top_p": req_top_p,
                "max_tokens": req_max_tokens,
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

            console.print("[bold cyan]LLM Streaming Started[/bold cyan]")

            full_text_parts: list[str] = []
            reasoning_parts: list[str] = []
            pending = ""
            mode: Literal["normal", "think"] = "normal"
            response_header_printed = False
            think_status: Status | None = None
            think_char_count = 0
            think_preview = ""
            last_status_update = 0.0
            status_update_interval_seconds = 0.2
            think_summary_pending = ""
            think_summary_mode: Literal["normal", "tag"] = "normal"
            think_summary_stream_parts: list[str] = []
            finish_reason: str | None = None

            def _start_think_status() -> None:
                nonlocal think_status
                if think_status is not None:
                    return
                console.print("\n[bold magenta]THINK (streaming)[/bold magenta]")
                think_status = console.status(
                    "[magenta]Thinking...[/magenta]",
                    spinner="dots",
                    spinner_style="magenta",
                )
                think_status.start()

            def _update_think_stream(chunk: str) -> None:
                nonlocal think_char_count, think_preview, last_status_update
                if not chunk:
                    return

                think_char_count += len(chunk)
                think_preview = (think_preview + chunk)[-80:]

                now = time.monotonic()
                if (now - last_status_update) < status_update_interval_seconds:
                    return

                _start_think_status()
                compact_preview = " ".join(think_preview.split())
                if think_status is not None:
                    think_status.update(
                        f"[magenta]Thinking... {think_char_count} chars[/magenta]"
                        f" [dim]{compact_preview}[/dim]"
                    )
                last_status_update = now

            def _stop_think_stream() -> None:
                nonlocal think_status
                if think_status is not None:
                    think_status.stop()
                    think_status = None

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

                        first_choice = choices[0]
                        if not isinstance(first_choice, dict):
                            continue

                        maybe_finish_reason = first_choice.get("finish_reason")
                        if isinstance(maybe_finish_reason, str) and maybe_finish_reason:
                            finish_reason = maybe_finish_reason

                        delta = first_choice.get("delta", {})
                        if not isinstance(delta, dict):
                            continue

                        reasoning_chunk = (
                            self._extract_delta_text(delta, "reasoning_content")
                            or self._extract_delta_text(delta, "reasoning")
                            or self._extract_delta_text(delta, "thinking")
                        )

                        if reasoning_chunk:
                            reasoning_parts.append(reasoning_chunk)
                            _update_think_stream(reasoning_chunk)

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
                            _update_think_stream(think_delta)

                        if normal_delta:
                            summary_delta, visible_delta, think_summary_pending, think_summary_mode = self._stream_split_by_tag(
                                chunk=normal_delta,
                                pending=think_summary_pending,
                                mode=think_summary_mode,
                                tag="think_summary",
                            )
                            if summary_delta:
                                think_summary_stream_parts.append(summary_delta)

                            if not visible_delta:
                                continue

                            _stop_think_stream()
                            if not response_header_printed:
                                console.print("\n[bold green]🎯 RESPONSE[/bold green]")
                                response_header_printed = True
                            console.print(Text(visible_delta, style="bold white"), end="")

            except HTTPError as exc:
                _stop_think_stream()
                detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
                raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
            except URLError as exc:
                _stop_think_stream()
                raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

            if pending:
                if mode == "think":
                    reasoning_parts.append(pending)
                    _update_think_stream(pending)
                else:
                    summary_delta, visible_delta, think_summary_pending, think_summary_mode = self._stream_split_by_tag(
                        chunk=pending,
                        pending=think_summary_pending,
                        mode=think_summary_mode,
                        tag="think_summary",
                    )
                    if summary_delta:
                        think_summary_stream_parts.append(summary_delta)
                    if not visible_delta:
                        visible_delta = ""

                    if visible_delta:
                        if not response_header_printed:
                            console.print("\n[bold green]🎯 RESPONSE[/bold green]")
                            response_header_printed = True
                        console.print(Text(visible_delta, style="bold white"), end="")

            if think_summary_pending:
                if think_summary_mode == "tag":
                    think_summary_stream_parts.append(think_summary_pending)
                elif think_summary_pending.strip():
                    if not response_header_printed:
                        console.print("\n[bold green]🎯 RESPONSE[/bold green]")
                        response_header_printed = True
                    console.print(Text(think_summary_pending, style="bold white"), end="")

            _stop_think_stream()

            console.print("\n[bold cyan]LLM Streaming Finished[/bold cyan]")

            content_text = "".join(full_text_parts)
            reasoning_text = "".join(reasoning_parts)
            think_summary_text = "".join(think_summary_stream_parts).strip()
            if not think_summary_text:
                think_summary_text = self._extract_tag_content(content_text, "think_summary")

            if think_summary_text:
                console.print("\n[bold magenta]THINK SUMMARY[/bold magenta]")
                console.print(Text(think_summary_text, style="bold magenta"))

            if finish_reason == "length":
                if attempt < retry_limit:
                    console.print(
                        f"[bold yellow]Output reached max_tokens={req_max_tokens}; retrying ({attempt + 1}/{retry_limit})[/bold yellow]"
                    )
                    continue
                raise RuntimeError(
                    f"LLM output exceeded max_tokens={req_max_tokens} after {retry_limit + 1} attempts."
                )

            if reasoning_text and "<think>" not in content_text:
                return f"<think>{reasoning_text}</think>\n{content_text}"
            return content_text

        raise RuntimeError("LLM request exhausted retry loop unexpectedly.")