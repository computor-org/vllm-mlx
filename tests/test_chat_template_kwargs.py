# SPDX-License-Identifier: Apache-2.0
"""Focused regressions for chat_template_kwargs forwarding."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import vllm_mlx.server as srv
from vllm_mlx.engine.base import GenerationOutput


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_chat_completion_request_preserves_chat_template_kwargs():
    request = srv.ChatCompletionRequest(
        model="test-model",
        messages=[srv.Message(role="user", content="Hello")],
        chat_template_kwargs={"enable_thinking": False},
    )

    assert request.chat_template_kwargs == {"enable_thinking": False}


def test_chat_completion_endpoint_forwards_chat_template_kwargs():
    captured = {}

    class FakeEngine:
        model_name = "test-model"
        is_mllm = False
        preserve_native_tool_format = False

        async def chat(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return GenerationOutput(
                text="ORBIT",
                prompt_tokens=4,
                completion_tokens=1,
                finish_reason="stop",
            )

    client = TestClient(srv.app)
    original_engine = srv._engine
    original_model_name = srv._model_name
    srv._engine = FakeEngine()
    srv._model_name = "test-model"
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Reply with ORBIT."}],
                "max_tokens": 8,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    finally:
        srv._engine = original_engine
        srv._model_name = original_model_name

    assert response.status_code == 200
    assert captured["kwargs"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert response.json()["choices"][0]["message"]["content"] == "ORBIT"


def test_llm_chat_applies_chat_template_kwargs_before_generate():
    from vllm_mlx.models.llm import MLXLanguageModel

    model = MLXLanguageModel.__new__(MLXLanguageModel)
    model._loaded = True
    model.tokenizer = MagicMock()
    model.tokenizer.apply_chat_template.return_value = "prompt"
    model.generate = MagicMock(return_value="ok")

    result = model.chat(
        [{"role": "user", "content": "Hello"}],
        chat_template_kwargs={"enable_thinking": False},
    )

    assert result == "ok"
    model.tokenizer.apply_chat_template.assert_called_once()
    assert (
        model.tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"] is False
    )
    model.generate.assert_called_once()


@pytest.mark.anyio
async def test_simple_engine_llm_chat_forwards_chat_template_kwargs():
    from vllm_mlx.engine.simple import SimpleEngine

    with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
        engine = SimpleEngine("test-model")
        engine._loaded = True
        engine._is_mllm = False
        engine._model = MagicMock()
        engine._model.chat = MagicMock(
            return_value=SimpleNamespace(
                text="OK",
                tokens=[1],
                finish_reason="stop",
            )
        )

        await engine.chat(
            [{"role": "user", "content": "Hello"}],
            chat_template_kwargs={"enable_thinking": False},
        )

        assert engine._model.chat.call_args.kwargs["chat_template_kwargs"] == {
            "enable_thinking": False
        }


@pytest.mark.anyio
async def test_simple_engine_tool_fallback_preserves_stream_state_and_kwargs():
    from vllm_mlx.engine.simple import SimpleEngine

    captured = {}

    async def fake_stream_chat(*args, **kwargs):
        captured["kwargs"] = kwargs
        yield SimpleNamespace(
            text="partial",
            tokens=[7],
            prompt_tokens=11,
            completion_tokens=1,
            finish_reason=None,
            finished=False,
        )
        yield SimpleNamespace(
            text="<|im_end|><tool_call>{\"name\":\"bash\",\"arguments\":{\"command\":\"pwd\"}}</tool_call>",
            tokens=[7, 8],
            prompt_tokens=11,
            completion_tokens=4,
            finish_reason="stop",
            finished=True,
        )

    with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
        engine = SimpleEngine("test-model")
        engine._loaded = True
        engine._is_mllm = False
        engine._model = MagicMock()
        engine._model.tokenizer.encode = MagicMock(return_value=[99])
        engine.stream_chat = fake_stream_chat  # type: ignore[method-assign]

        output = await engine.chat(
            [{"role": "user", "content": "Hello"}],
            tools=[{"type": "function", "function": {"name": "bash"}}],
            chat_template_kwargs={"enable_thinking": False},
        )

        assert captured["kwargs"]["chat_template_kwargs"] == {
            "enable_thinking": False
        }
        assert output.tokens == [7, 8]
        assert output.prompt_tokens == 11
        assert output.completion_tokens == 4
        assert output.finish_reason == "stop"
