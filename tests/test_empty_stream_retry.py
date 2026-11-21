"""流式与非流式空回复缓冲、诊断与重试回归测试。"""
import asyncio
import json

from fastapi import Response

from src.api import antigravity

async def _no_op(*args, **kwargs):
    return None


async def _false_async(*args, **kwargs):
    return False


def _patch_stream_deps(monkeypatch, *, creds, stream_fn, max_retries=0):
    monkeypatch.setattr(antigravity, "get_antigravity_api_url", lambda: _const("https://example.invalid"))
    monkeypatch.setattr(antigravity, "get_auto_ban_error_codes", lambda: _const([403]))
    monkeypatch.setattr(
        antigravity,
        "get_retry_config",
        lambda: _const({"retry_enabled": True, "max_retries": max_retries, "retry_interval": 0.0}),
    )
    monkeypatch.setattr(antigravity, "get_request_log_enabled", _false_async)
    monkeypatch.setattr(antigravity, "record_usage_log", _no_op)

    success_calls = []
    error_calls = []

    async def _rec_ok(email, model_name=None):
        success_calls.append((email, model_name))

    async def _rec_err(email, status_code, cooldown_until=None, model_name=None, error_message=None):
        error_calls.append((email, status_code, error_message))

    monkeypatch.setattr(antigravity, "record_api_call_success", _rec_ok)
    monkeypatch.setattr(antigravity, "record_api_call_error", _rec_err)

    async def _get_cred(model_name=None, session_key=None, exclude_emails=None):
        exclude = exclude_emails or set()
        for email, data in creds:
            if email not in exclude:
                return email, data
        return None

    monkeypatch.setattr(antigravity, "get_valid_credential", _get_cred)
    monkeypatch.setattr(antigravity, "stream_post_async", stream_fn)
    return success_calls, error_calls


async def _const(val):
    return val


async def test_stream_empty_shell_retried_and_recovers(monkeypatch):
    """首试返回仅有 usageMetadata 的空壳流时，缓冲不提前成功，换凭证并重试后正常输出。"""
    seen_calls = []

    async def fake_stream_post(url, body, native=False, headers=None):
        seen_calls.append(
            {
                "project": body["project"],
                "requestId": body["requestId"],
                "sessionId": body["request"]["sessionId"],
            }
        )
        if len(seen_calls) == 1:
            # 模拟上游返回 200 SSE 但 candidates 为空/无 text，只有 usageMetadata
            yield 'data: {"response":{"candidates":[{"content":{"role":"model","parts":[]},"index":0}],"usageMetadata":{"promptTokenCount":25000,"candidatesTokenCount":0}}}'
            yield ""
            return
        yield 'data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"你好"}]},"finishReason":"STOP","index":0}]}}'

    success_calls, error_calls = _patch_stream_deps(
        monkeypatch,
        creds=[
            ("a@example.com", {"access_token": "tok-a", "project_id": "proj-a"}),
            ("b@example.com", {"access_token": "tok-b", "project_id": "proj-b"}),
        ],
        stream_fn=fake_stream_post,
        max_retries=0,  # 即使 429 重试为 0，空流仍应重试
    )

    chunks = []
    async for item in antigravity.stream_request(
        body={"model": "gemini-4-hydrogen-high", "request": {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}},
        native=False,
    ):
        chunks.append(item)

    assert len(seen_calls) == 2
    assert seen_calls[0]["project"] == "proj-a"
    assert seen_calls[1]["project"] == "proj-b"
    assert seen_calls[0]["requestId"] != seen_calls[1]["requestId"]
    assert len(chunks) == 1
    assert '"你好"' in chunks[0]
    assert success_calls == [("b@example.com", "gemini-4-hydrogen-high")]
    assert len(error_calls) == 1
    assert error_calls[0][0] == "a@example.com"
    assert "Empty response from API" in error_calls[0][2]


async def test_stream_all_empty_rotates_session_and_returns_502(monkeypatch):
    """连续空流时第 2 次重试轮换 sessionId/trajectory_id，耗尽后返回 502 Response。"""
    seen_calls = []

    async def fake_always_empty(url, body, native=False, headers=None):
        seen_calls.append(
            {
                "requestId": body["requestId"],
                "sessionId": body["request"]["sessionId"],
                "trajectory": body["request"]["labels"]["trajectory_id"],
            }
        )
        yield 'data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":""}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":30000}}}'

    success_calls, error_calls = _patch_stream_deps(
        monkeypatch,
        creds=[("a@example.com", {"access_token": "tok-a", "project_id": "proj-a"})],
        stream_fn=fake_always_empty,
        max_retries=0,
    )

    chunks = []
    async for item in antigravity.stream_request(
        body={"model": "gemini-4-hydrogen-high", "request": {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}},
        native=False,
    ):
        chunks.append(item)

    # 初试 + 2 次空流重试 = 3 次
    assert len(seen_calls) == 3
    assert seen_calls[0]["sessionId"] == seen_calls[1]["sessionId"]
    assert seen_calls[2]["sessionId"] != seen_calls[0]["sessionId"]
    assert seen_calls[2]["trajectory"] != seen_calls[0]["trajectory"]
    assert len(chunks) == 1
    assert isinstance(chunks[0], Response)
    assert chunks[0].status_code == 502
    assert "upstream_empty_response" in chunks[0].body.decode("utf-8")
    assert success_calls == []
    assert len(error_calls) == 3


async def test_stream_buffers_empty_prefix_then_flushes_on_real_text(monkeypatch):
    """首块为空元数据、次块有真实文本时，缓冲首块并在见到文本后完整透传（含首块）。"""
    async def fake_prefix_then_text(url, body, native=False, headers=None):
        yield 'data: {"response":{"candidates":[{"content":{"role":"model","parts":[]},"index":0}]}}'
        yield 'data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP","index":0}]}}'

    success_calls, error_calls = _patch_stream_deps(
        monkeypatch,
        creds=[("a@example.com", {"access_token": "tok-a", "project_id": "proj-a"})],
        stream_fn=fake_prefix_then_text,
        max_retries=0,
    )

    chunks = []
    async for item in antigravity.stream_request(
        body={"model": "gemini-4-hydrogen-high", "request": {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}},
        native=False,
    ):
        chunks.append(item)

    assert len(chunks) == 2
    assert '"ok"' in chunks[1]
    assert success_calls == [("a@example.com", "gemini-4-hydrogen-high")]
    assert error_calls == []


async def test_non_stream_empty_json_shell_retries_and_returns_502(monkeypatch):
    """非流式 200 仅返回无内容 JSON 时同样触发重试，耗尽返回 502。"""
    monkeypatch.setattr(antigravity, "get_antigravity_api_url", lambda: _const("https://example.invalid"))
    monkeypatch.setattr(antigravity, "get_auto_ban_error_codes", lambda: _const([403]))
    monkeypatch.setattr(
        antigravity,
        "get_retry_config",
        lambda: _const({"retry_enabled": True, "max_retries": 0, "retry_interval": 0.0}),
    )
    monkeypatch.setattr(antigravity, "get_request_log_enabled", _false_async)
    monkeypatch.setattr(antigravity, "record_usage_log", _no_op)
    monkeypatch.setattr(antigravity, "record_api_call_success", _no_op)
    monkeypatch.setattr(antigravity, "record_api_call_error", _no_op)
    monkeypatch.setattr(
        antigravity,
        "get_valid_credential",
        lambda **kwargs: _const(("a@example.com", {"access_token": "tok-a", "project_id": "proj-a"})),
    )

    class FakeHttpResp:
        status_code = 200
        headers = {"content-type": "application/json"}
        content = json.dumps({"response": {"candidates": [{"content": {"role": "model", "parts": []}}]}}).encode()
        text = content.decode()

    calls = 0

    async def fake_post(url, json=None, headers=None):
        nonlocal calls
        calls += 1
        return FakeHttpResp()

    monkeypatch.setattr(antigravity, "post_async", fake_post)

    resp = await antigravity.non_stream_request(
        body={"model": "gemini-4-hydrogen-high", "request": {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}},
    )
    assert calls == 3
    assert resp.status_code == 502
