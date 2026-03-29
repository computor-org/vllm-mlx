# SPDX-License-Identifier: Apache-2.0
"""Tests that tool-enabled chat preserves raw parser-visible output."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSimpleEngineToolOutputPreservation:
    @pytest.mark.anyio
    async def test_chat_with_tools_preserves_raw_harmony_output(self):
        from vllm_mlx.engine.simple import SimpleEngine

        async def fake_stream_chat(*args, **kwargs):
            yield MagicMock(
                text=(
                    "<|channel|>commentary to=functions.get_weather"
                    '<|message|>{"city":"Paris"}<|call|>'
                ),
                tokens=[],
                prompt_tokens=11,
                completion_tokens=4,
                finish_reason="stop",
                finished=True,
            )

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = MagicMock()
            engine._loaded = True
            engine.stream_chat = fake_stream_chat  # type: ignore[method-assign]

            output = await engine.chat(
                messages=[{"role": "user", "content": "Weather in Paris?"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )

            assert "<|channel|>commentary" in output.text
            assert "<|call|>" in output.text


class TestBatchedEngineToolOutputPreservation:
    @pytest.mark.anyio
    async def test_chat_with_tools_preserves_raw_output(self):
        from vllm_mlx.engine.batched import BatchedEngine

        raw_output = (
            "<|channel|>commentary to=functions.get_weather"
            '<|message|>{"city":"Paris"}<|call|>'
        )

        with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=False):
            engine = BatchedEngine("test-model")
            engine._loaded = True
            engine._tokenizer = MagicMock()
            engine._apply_chat_template = MagicMock(return_value="prompt")
            engine._engine = MagicMock()
            engine._engine.generate = AsyncMock(
                return_value=MagicMock(
                    output_text=raw_output,
                    prompt_tokens=9,
                    completion_tokens=3,
                    finish_reason="stop",
                )
            )

            output = await engine.chat(
                messages=[{"role": "user", "content": "Weather in Paris?"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )

            assert output.text == raw_output
            engine._engine.generate.assert_called_once()
