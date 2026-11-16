from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


class _Retryable(Exception):
    def __init__(self, reason: str, retry_after: float | None = None, rate_limited: bool = False):
        super().__init__(reason)
        self.retry_after = retry_after
        self.rate_limited = rate_limited


def load_env_file(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.removeprefix("export ").split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass(frozen=True)
class LLMConfig:
    model: str = "qwen3"
    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens: int | None = 8192
    enable_thinking: bool | None = None
    reasoning_effort: str | None = None
    stream: bool = True
    connect_timeout: float = 15.0
    idle_timeout: float = 120.0
    total_timeout: float = 900.0
    slow_ttft: float = 60.0
    max_concurrency: int = 8
    max_attempts: int = 8
    patience: float = float(os.environ.get("GRAPH_MEMORY_LLM_PATIENCE", "600"))
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str

    def parsed_args(self) -> dict[str, Any] | None:
        try:
            value = json.loads(self.arguments or "{}")
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None


@dataclass
class ChatResult:
    content: str
    reasoning: str
    tool_calls: list[ToolCall]
    finish_reason: str | None
    usage: dict[str, Any]
    model: str | None
    latency: float
    ttft: float | None
    attempts: int = 1


class AdaptiveLimiter:

    def __init__(self, max_limit: int):
        self.max_limit = max_limit
        self.limit = max_limit
        self.in_flight = 0
        self._streak = 0
        self._cond = asyncio.Condition()

    async def __aenter__(self) -> None:
        async with self._cond:
            await self._cond.wait_for(lambda: self.in_flight < self.limit)
            self.in_flight += 1

    async def __aexit__(self, *exc: Any) -> None:
        async with self._cond:
            self.in_flight -= 1
            self._cond.notify_all()

    async def congested(self) -> None:
        async with self._cond:
            self.limit = max(1, self.limit // 2)
            self._streak = 0

    async def succeeded(self) -> None:
        async with self._cond:
            self._streak += 1
            if self.limit < self.max_limit and self._streak >= 4 * self.limit:
                self.limit += 1
                self._streak = 0
                self._cond.notify_all()


class LLMClient:
    def __init__(
        self,
        cfg: LLMConfig = LLMConfig(),
        *,
        env_file: str | Path = ".env",
        log_path: str | Path | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        env = {**load_env_file(env_file), **{k: v for k, v in os.environ.items() if k.startswith("OPENAI_")}}
        api_key = api_key or env.get("OPENAI_API_KEY")
        base_url = base_url or env.get("OPENAI_BASE_URL")
        if not api_key or not base_url:
            raise LLMError("OPENAI_API_KEY / OPENAI_BASE_URL missing (set them in the environment or in .env)")
        self.cfg = cfg
        self.base_url = base_url.rstrip("/").removesuffix("/chat/completions")
        self.limiter = AdaptiveLimiter(cfg.max_concurrency)
        self.log_path = Path(log_path) if log_path else None
        self.models_seen: set[str] = set()
        self.totals = {"calls": 0, "retries": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._http = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(cfg.total_timeout, connect=cfg.connect_timeout, read=cfg.idle_timeout),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    def _payload(self, messages: list[dict], tools: list[dict] | None, cfg: LLMConfig) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "stream": cfg.stream,
        }
        if cfg.max_tokens:
            payload["max_tokens"] = cfg.max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if cfg.stream:
            payload["stream_options"] = {"include_usage": True}
        if cfg.enable_thinking is False:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        elif cfg.enable_thinking and cfg.reasoning_effort:
            payload["reasoning_effort"] = cfg.reasoning_effort
        payload.update(cfg.extra_body)
        return payload

    async def chat(self, messages: list[dict], tools: list[dict] | None = None, **overrides: Any) -> ChatResult:
        cfg = replace(self.cfg, **overrides) if overrides else self.cfg
        payload = self._payload(messages, tools, cfg)
        errors: list[str] = []
        start, attempt = time.monotonic(), 0
        while True:
            attempt += 1
            retry: _Retryable | None = None
            async with self.limiter:
                try:
                    async with asyncio.timeout(cfg.total_timeout):
                        once = self._stream_once if cfg.stream else self._post_once
                        result = await once(payload)
                except _Retryable as exc:
                    retry = exc
                except (httpx.TimeoutException, TimeoutError):
                    retry = _Retryable("timeout")
                except httpx.TransportError as exc:
                    retry = _Retryable(f"transport:{type(exc).__name__}")
                except (json.JSONDecodeError, KeyError, IndexError) as exc:
                    retry = _Retryable(f"malformed response: {type(exc).__name__}")
            if retry is None:
                result.attempts = attempt
                await self._record(cfg, result, errors)
                if result.ttft is not None and result.ttft > cfg.slow_ttft:
                    await self.limiter.congested()
                else:
                    await self.limiter.succeeded()
                return result
            errors = (errors + [str(retry)])[-20:]
            self.totals["retries"] += 1
            if retry.rate_limited or str(retry) == "timeout":
                await self.limiter.congested()
            if not self._keep_trying(attempt, start, cfg, errors):
                break
            delay = retry.retry_after or min(60.0, 2.0 ** min(attempt, 6)) * (0.5 + random.random())
            await asyncio.sleep(delay)
        self._log({"event": "chat_failed", "model": cfg.model, "errors": errors})
        raise LLMError(f"chat failed after {attempt} attempts: {errors[-3:]}")

    def _keep_trying(self, attempt: int, start: float, cfg: LLMConfig, errors: list[str]) -> bool:
        if attempt < cfg.max_attempts:
            return True
        if time.monotonic() - start < cfg.patience:
            if attempt == cfg.max_attempts:
                self._log({"event": "chat_waiting_out_outage", "errors": errors[-3:], "patience_s": cfg.patience})
            return True
        return False

    async def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code == 200:
            return
        body = (await resp.aread())[:300].decode("utf-8", "replace")
        if resp.status_code in RETRYABLE_STATUS:
            retry_after = resp.headers.get("retry-after")
            try:
                retry_after_s = float(retry_after) if retry_after else None
            except ValueError:
                retry_after_s = None
            raise _Retryable(f"http {resp.status_code}", retry_after_s, rate_limited=resp.status_code == 429)
        raise LLMError(f"http {resp.status_code}: {body}")

    async def _stream_once(self, payload: dict[str, Any]) -> ChatResult:
        start = time.monotonic()
        ttft = None
        content: list[str] = []
        reasoning: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        finish = None
        usage: dict[str, Any] = {}
        model = None
        async with self._http.stream("POST", f"{self.base_url}/chat/completions", json=payload) as resp:
            await self._raise_for_status(resp)
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    raise _Retryable(f"stream error: {str(chunk['error'])[:200]}")
                model = chunk.get("model") or model
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("reasoning_content") or delta.get("reasoning")
                    if piece:
                        reasoning.append(piece)
                    if delta.get("content"):
                        content.append(delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        slot = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                        fn = tc.get("function") or {}
                        slot["id"] = slot["id"] or tc.get("id") or ""
                        slot["name"] = slot["name"] or fn.get("name") or ""
                        slot["arguments"] += fn.get("arguments") or ""
                    if ttft is None and (piece or delta.get("content") or delta.get("tool_calls")):
                        ttft = time.monotonic() - start
                    finish = choice.get("finish_reason") or finish
        tool_calls = [
            ToolCall(id=s["id"] or f"call_{i}", name=s["name"], arguments=s["arguments"])
            for i, s in sorted(calls.items())
        ]
        return ChatResult("".join(content), "".join(reasoning), tool_calls, finish, usage, model,
                          time.monotonic() - start, ttft)

    async def _post_once(self, payload: dict[str, Any]) -> ChatResult:
        start = time.monotonic()
        resp = await self._http.post(f"{self.base_url}/chat/completions", json=payload)
        await self._raise_for_status(resp)
        data = resp.json()
        choice = data["choices"][0]
        message = choice.get("message") or {}
        tool_calls = [
            ToolCall(id=tc.get("id") or f"call_{i}", name=tc["function"]["name"],
                     arguments=tc["function"].get("arguments") or "")
            for i, tc in enumerate(message.get("tool_calls") or [])
        ]
        latency = time.monotonic() - start
        return ChatResult(message.get("content") or "",
                          message.get("reasoning_content") or message.get("reasoning") or "",
                          tool_calls, choice.get("finish_reason"), data.get("usage") or {},
                          data.get("model"), latency, None)

    async def _record(self, cfg: LLMConfig, result: ChatResult, errors: list[str]) -> None:
        self.totals["calls"] += 1
        self.totals["prompt_tokens"] += int(result.usage.get("prompt_tokens") or 0)
        self.totals["completion_tokens"] += int(result.usage.get("completion_tokens") or 0)
        if result.model:
            if self.models_seen and result.model not in self.models_seen:
                self._log({"event": "model_changed", "previous": sorted(self.models_seen), "now": result.model})
            self.models_seen.add(result.model)
        self._log({
            "event": "chat",
            "requested_model": cfg.model,
            "response_model": result.model,
            "thinking": cfg.enable_thinking,
            "latency": round(result.latency, 3),
            "ttft": None if result.ttft is None else round(result.ttft, 3),
            "attempts": result.attempts,
            "retry_errors": errors,
            "finish_reason": result.finish_reason,
            "usage": result.usage,
            "n_tool_calls": len(result.tool_calls),
        })

    def _log(self, record: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": time.time(), **record}
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
