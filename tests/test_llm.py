import json

import httpx
import pytest

from graph_memory.llm import LLMClient, LLMConfig, LLMError


def sse(*chunks: dict) -> bytes:
    return b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks) + b"data: [DONE]\n\n"


def client_with(handler, **cfg) -> LLMClient:
    return LLMClient(LLMConfig(**({"max_attempts": 3, "patience": 0.0} | cfg)), env_file="/nonexistent", api_key="k",
                     base_url="https://example.test/v1", transport=httpx.MockTransport(handler))


STREAM = sse(
    {"model": "m1", "choices": [{"delta": {"reasoning_content": "think "}}]},
    {"model": "m1", "choices": [{"delta": {"reasoning_content": "more"}}]},
    {"model": "m1", "choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "c1", "function": {"name": "act", "arguments": '{"act'}}]}}]},
    {"model": "m1", "choices": [{"delta": {"tool_calls": [
        {"index": 0, "function": {"arguments": 'ion": "look"}'}}]}, "finish_reason": "tool_calls"}]},
    {"model": "m1", "choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
)


async def test_stream_assembles_tool_call_and_reasoning():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=STREAM)

    async with client_with(handler, enable_thinking=True, reasoning_effort="low") as client:
        res = await client.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    assert res.reasoning == "think more"
    assert [(c.id, c.name, c.parsed_args()) for c in res.tool_calls] == [("c1", "act", {"action": "look"})]
    assert res.finish_reason == "tool_calls" and res.usage["completion_tokens"] == 5 and res.model == "m1"
    assert seen["body"]["reasoning_effort"] == "low" and seen["body"]["stream"] is True


async def test_thinking_switch():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=STREAM)

    async with client_with(handler, enable_thinking=False) as client:
        await client.chat([{"role": "user", "content": "hi"}])
    async with client_with(handler) as client:
        await client.chat([{"role": "user", "content": "hi"}])
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False} and "reasoning_effort" not in seen[0]
    assert "chat_template_kwargs" not in seen[1] and "reasoning_effort" not in seen[1]


async def test_retries_429_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0.01"})
        return httpx.Response(200, content=STREAM)

    async with client_with(handler, max_concurrency=4) as client:
        res = await client.chat([{"role": "user", "content": "hi"}])
        assert res.attempts == 2 and client.limiter.limit == 2


async def test_non_retryable_status_raises():
    async with client_with(lambda r: httpx.Response(400, text="bad")) as client:
        with pytest.raises(LLMError):
            await client.chat([{"role": "user", "content": "hi"}])


async def test_outage_is_waited_out_within_patience_then_given_up():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] <= 5:
            return httpx.Response(503, headers={"retry-after": "0.001"})
        return httpx.Response(200, content=STREAM)

    async with client_with(handler, max_attempts=2, patience=60.0) as client:
        await client.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 6
    calls["n"] = 0
    async with client_with(handler, max_attempts=2) as client:
        with pytest.raises(LLMError):
            await client.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 2


def test_missing_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(LLMError):
        LLMClient(env_file="/nonexistent")
