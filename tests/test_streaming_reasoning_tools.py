# SPDX-License-Identifier: Apache-2.0
"""Streaming chat-completion tests for reasoning + tool parser coexistence."""

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

import vllm_mlx.server as server
from vllm_mlx.api.models import ChatCompletionRequest, Message, ToolDefinition
from vllm_mlx.engine.base import BaseEngine, GenerationOutput
from vllm_mlx.reasoning import get_parser
from vllm_mlx.tool_parsers import ToolParserManager


TEST_TOOL = ToolDefinition(
    function={
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
)


class FakeStreamEngine(BaseEngine):
    """Minimal engine for deterministic stream_chat tests."""

    def __init__(self, deltas: list[str], model_name: str = "test-model"):
        self._deltas = deltas
        self._model_name = model_name
        self._tokenizer = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_mllm(self) -> bool:
        return False

    @property
    def tokenizer(self) -> Any:
        return None

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def generate(self, *args, **kwargs) -> GenerationOutput:
        raise NotImplementedError

    async def stream_generate(self, *args, **kwargs) -> AsyncIterator[GenerationOutput]:
        raise NotImplementedError

    async def chat(self, *args, **kwargs) -> GenerationOutput:
        raise NotImplementedError

    async def stream_chat(self, *args, **kwargs) -> AsyncIterator[GenerationOutput]:
        text = ""
        for i, delta in enumerate(self._deltas):
            text += delta
            yield GenerationOutput(
                text=text,
                new_text=delta,
                finished=i == len(self._deltas) - 1,
                completion_tokens=i + 1,
                finish_reason="stop",
            )


def _collect_payloads(stream_output: list[str]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for chunk in stream_output:
        assert chunk.startswith("data: ")
        payload = chunk[6:].strip()
        if payload == "[DONE]":
            continue
        payloads.append(json.loads(payload))
    return payloads


def _flatten_deltas(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deltas: list[dict[str, Any]] = []
    for payload in payloads:
        for choice in payload["choices"]:
            deltas.append(
                {
                    "delta": choice["delta"],
                    "finish_reason": choice["finish_reason"],
                }
            )
    return deltas


def _reasoning_text(deltas: list[dict[str, Any]]) -> str:
    return "".join(delta["delta"].get("reasoning") or "" for delta in deltas)


def _content_text(deltas: list[dict[str, Any]]) -> str:
    return "".join(delta["delta"].get("content") or "" for delta in deltas)


def _tool_calls(deltas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for delta in deltas:
        calls.extend(delta["delta"].get("tool_calls") or [])
    return calls


async def _run_stream(
    monkeypatch: pytest.MonkeyPatch,
    *,
    deltas: list[str],
    reasoning_parser: str | None,
    tool_parser: str | None,
    model_name: str = "test-model",
) -> list[dict[str, Any]]:
    engine = FakeStreamEngine(deltas, model_name=model_name)
    parser_instance = (
        ToolParserManager.get_tool_parser(tool_parser)() if tool_parser else None
    )

    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(server, "_model_name", model_name)
    monkeypatch.setattr(
        server,
        "_reasoning_parser",
        get_parser(reasoning_parser)() if reasoning_parser else None,
    )
    monkeypatch.setattr(server, "_enable_auto_tool_choice", tool_parser is not None)
    monkeypatch.setattr(server, "_tool_call_parser", tool_parser)
    monkeypatch.setattr(server, "_tool_parser_instance", parser_instance)

    request = ChatCompletionRequest(
        model=model_name,
        messages=[Message(role="user", content="Weather in Paris?")],
        stream=True,
        tools=[TEST_TOOL],
    )

    raw_chunks: list[str] = []
    async for chunk in server.stream_chat_completion(
        engine,
        [{"role": "user", "content": "Weather in Paris?"}],
        request,
    ):
        raw_chunks.append(chunk)

    return _flatten_deltas(_collect_payloads(raw_chunks))


@pytest.mark.anyio
async def test_streaming_qwen_reasoning_then_tool_call(monkeypatch):
    deltas = [
        "<think>",
        "Need tool",
        "</think>",
        "<tool_call>\n",
        '{"name":"get_weather","arguments":{"city":"Paris"}}',
        "\n</tool_call>",
    ]

    deltas_out = await _run_stream(
        monkeypatch,
        deltas=deltas,
        reasoning_parser="qwen3",
        tool_parser="qwen",
    )

    calls = _tool_calls(deltas_out)

    assert _reasoning_text(deltas_out) == "Need tool"
    assert _content_text(deltas_out) == ""
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
    assert deltas_out[-1]["finish_reason"] == "tool_calls"


@pytest.mark.anyio
async def test_streaming_qwen_direct_tool_call_with_reasoning_parser(monkeypatch):
    deltas = [
        "<tool_call>\n",
        '{"name":"get_weather","arguments":{"city":"Paris"}}',
        "\n</tool_call>",
    ]

    deltas_out = await _run_stream(
        monkeypatch,
        deltas=deltas,
        reasoning_parser="qwen3",
        tool_parser="qwen",
    )

    calls = _tool_calls(deltas_out)

    assert _reasoning_text(deltas_out) == ""
    assert _content_text(deltas_out) == ""
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert deltas_out[-1]["finish_reason"] == "tool_calls"


@pytest.mark.anyio
async def test_streaming_qwen_reasoning_then_plain_text(monkeypatch):
    deltas = ["<think>", "R", "</think>", "Hello", " world"]

    deltas_out = await _run_stream(
        monkeypatch,
        deltas=deltas,
        reasoning_parser="qwen3",
        tool_parser="qwen",
    )

    assert _reasoning_text(deltas_out) == "R"
    assert _content_text(deltas_out) == "Hello world"
    assert _tool_calls(deltas_out) == []
    assert deltas_out[-1]["finish_reason"] == "stop"


@pytest.mark.anyio
async def test_streaming_qwen_tool_call_inside_think_is_not_emitted(monkeypatch):
    deltas = [
        "<think>",
        'R <tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}',
        "</tool_call></think>",
        "Text",
    ]

    deltas_out = await _run_stream(
        monkeypatch,
        deltas=deltas,
        reasoning_parser="qwen3",
        tool_parser="qwen",
    )

    assert "<tool_call>" in _reasoning_text(deltas_out)
    assert _content_text(deltas_out) == "Text"
    assert _tool_calls(deltas_out) == []
    assert deltas_out[-1]["finish_reason"] == "stop"


@pytest.mark.anyio
async def test_streaming_harmony_reasoning_then_tool_call(monkeypatch):
    deltas = [
        "<|channel|>analysis",
        "<|message|>",
        "Need weather",
        "<|end|>",
        "<|channel|>commentary to=functions.get_weather",
        "<|constrain|>json",
        "<|message|>",
        '{"city":"Paris"}',
        "<|call|>",
    ]

    deltas_out = await _run_stream(
        monkeypatch,
        deltas=deltas,
        reasoning_parser="harmony",
        tool_parser="harmony",
    )

    calls = _tool_calls(deltas_out)

    assert _reasoning_text(deltas_out) == "Need weather"
    assert _content_text(deltas_out) == ""
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
    assert deltas_out[-1]["finish_reason"] == "tool_calls"


@pytest.mark.anyio
async def test_streaming_gpt_oss_split_commentary_routes_to_harmony_tool_parser(
    monkeypatch,
):
    deltas = [
        "<|channel|>",
        "analysis",
        "<|message|>",
        "Need weather",
        "<|end|>",
        "<|channel|>",
        "comment",
        "ary to",
        "=functions.get_weather",
        " <|constrain|>",
        "json",
        "<|message|>",
        '{"city":"Paris"}',
        "<|call|>",
    ]

    deltas_out = await _run_stream(
        monkeypatch,
        deltas=deltas,
        reasoning_parser="gpt_oss",
        tool_parser="harmony",
    )

    calls = _tool_calls(deltas_out)

    assert _reasoning_text(deltas_out) == "Need weather"
    assert _content_text(deltas_out) == ""
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
    assert deltas_out[-1]["finish_reason"] == "tool_calls"


@pytest.mark.anyio
async def test_streaming_qwen_function_parameter_tool_call(monkeypatch):
    deltas = [
        "<think>",
        "Need tool",
        "</think>",
        "<tool_call>\n",
        "<function=get_weather>\n",
        "<parameter=city>\n",
        '"Paris"\n',
        "</parameter>\n",
        "<parameter=days>\n",
        "3\n",
        "</parameter>\n",
        "</function>\n",
        "</tool_call>",
    ]

    deltas_out = await _run_stream(
        monkeypatch,
        deltas=deltas,
        reasoning_parser="qwen3",
        tool_parser="qwen",
    )

    calls = _tool_calls(deltas_out)

    assert _reasoning_text(deltas_out) == "Need tool"
    assert _content_text(deltas_out) == ""
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "city": "Paris",
        "days": 3,
    }
    assert deltas_out[-1]["finish_reason"] == "tool_calls"


@pytest.mark.anyio
async def test_streaming_without_reasoning_parser_keeps_qwen_tools_working(monkeypatch):
    deltas = [
        "<tool_call>\n",
        '{"name":"get_weather","arguments":{"city":"Paris"}}',
        "\n</tool_call>",
    ]

    deltas_out = await _run_stream(
        monkeypatch,
        deltas=deltas,
        reasoning_parser=None,
        tool_parser="qwen",
    )

    calls = _tool_calls(deltas_out)

    assert _reasoning_text(deltas_out) == ""
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert deltas_out[-1]["finish_reason"] == "tool_calls"


@pytest.mark.anyio
async def test_streaming_without_reasoning_parser_keeps_mistral_tools_working(
    monkeypatch,
):
    deltas = ["[TOOL_CALLS]", 'get_weather{"city":"Paris"}']

    deltas_out = await _run_stream(
        monkeypatch,
        deltas=deltas,
        reasoning_parser=None,
        tool_parser="mistral",
    )

    calls = _tool_calls(deltas_out)

    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert deltas_out[-1]["finish_reason"] == "tool_calls"
