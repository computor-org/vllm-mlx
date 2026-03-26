# SPDX-License-Identifier: Apache-2.0
"""Tests for tool_choice enforcement across API endpoints."""

import json

import pytest
from fastapi.testclient import TestClient

import vllm_mlx.server as srv
from vllm_mlx.engine.base import GenerationOutput

TOOL_CALL_MARKUP = (
    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
)

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current time in a timezone",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
                "required": ["timezone"],
            },
        },
    },
]


class FakeEngine:
    """Fake engine that returns canned output containing tool call markup."""

    model_name = "test-model"
    is_mllm = False
    preserve_native_tool_format = False

    def __init__(self, text: str = TOOL_CALL_MARKUP):
        self._text = text
        self.captured_messages = None
        self.captured_kwargs = None

    async def chat(self, messages, **kwargs):
        self.captured_messages = messages
        self.captured_kwargs = kwargs
        return GenerationOutput(
            text=self._text,
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="stop",
        )


def _patch_engine(engine):
    """Context-manager-like helper to swap the global engine."""
    original_engine = srv._engine
    original_model = srv._model_name
    srv._engine = engine
    srv._model_name = "test-model"
    return original_engine, original_model


def _restore_engine(original_engine, original_model):
    srv._engine = original_engine
    srv._model_name = original_model


# ---------------------------------------------------------------------------
# Unit tests for _apply_tool_choice
# ---------------------------------------------------------------------------


class TestApplyToolChoice:
    """Direct unit tests for the _apply_tool_choice helper."""

    def test_none_strips_tools_and_returns_false(self):
        chat_kwargs = {"tools": [{"function": {"name": "f"}}]}
        messages = [{"role": "user", "content": "hi"}]
        result = srv._apply_tool_choice("none", chat_kwargs, messages)
        assert result is False
        assert "tools" not in chat_kwargs
        assert len(messages) == 1  # no system message added

    def test_required_adds_system_message(self):
        chat_kwargs = {"tools": [{"function": {"name": "f"}}]}
        messages = [{"role": "user", "content": "hi"}]
        result = srv._apply_tool_choice("required", chat_kwargs, messages)
        assert result is True
        assert len(messages) == 2
        assert messages[-1]["role"] == "system"
        assert "MUST call" in messages[-1]["content"]

    def test_dict_filters_tools_and_adds_system_message(self):
        chat_kwargs = {
            "tools": [
                {"function": {"name": "get_weather"}},
                {"function": {"name": "get_time"}},
            ]
        }
        messages = [{"role": "user", "content": "hi"}]
        result = srv._apply_tool_choice(
            {"function": {"name": "get_weather"}}, chat_kwargs, messages
        )
        assert result is True
        assert len(chat_kwargs["tools"]) == 1
        assert chat_kwargs["tools"][0]["function"]["name"] == "get_weather"
        assert len(messages) == 2
        assert "get_weather" in messages[-1]["content"]

    def test_dict_with_no_matching_tool_keeps_all(self):
        chat_kwargs = {
            "tools": [
                {"function": {"name": "get_weather"}},
            ]
        }
        messages = [{"role": "user", "content": "hi"}]
        srv._apply_tool_choice(
            {"function": {"name": "nonexistent"}}, chat_kwargs, messages
        )
        assert len(chat_kwargs["tools"]) == 1  # no filter applied

    def test_auto_returns_true_no_changes(self):
        chat_kwargs = {"tools": [{"function": {"name": "f"}}]}
        messages = [{"role": "user", "content": "hi"}]
        result = srv._apply_tool_choice("auto", chat_kwargs, messages)
        assert result is True
        assert "tools" in chat_kwargs
        assert len(messages) == 1

    def test_none_value_returns_true_no_changes(self):
        chat_kwargs = {"tools": [{"function": {"name": "f"}}]}
        messages = [{"role": "user", "content": "hi"}]
        result = srv._apply_tool_choice(None, chat_kwargs, messages)
        assert result is True
        assert "tools" in chat_kwargs
        assert len(messages) == 1


# ---------------------------------------------------------------------------
# Integration tests via the OpenAI chat endpoint
# ---------------------------------------------------------------------------


class TestToolChoiceOpenAIEndpoint:
    """Integration tests hitting /v1/chat/completions with tool_choice."""

    def test_tool_choice_none_strips_tools_and_skips_parsing(self):
        engine = FakeEngine(text=TOOL_CALL_MARKUP)
        orig_engine, orig_model = _patch_engine(engine)
        client = TestClient(srv.app)
        try:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "weather?"}],
                    "tools": SAMPLE_TOOLS,
                    "tool_choice": "none",
                    "max_tokens": 64,
                },
            )
        finally:
            _restore_engine(orig_engine, orig_model)

        assert response.status_code == 200
        data = response.json()
        msg = data["choices"][0]["message"]
        # tool_calls must be absent or None
        assert msg.get("tool_calls") is None
        # tools should have been stripped from kwargs sent to engine
        assert "tools" not in engine.captured_kwargs
        # The raw markup should appear as content since parsing was skipped
        assert "tool_call" in (msg.get("content") or "")

    def test_tool_choice_required_injects_system_message(self):
        engine = FakeEngine(text=TOOL_CALL_MARKUP)
        orig_engine, orig_model = _patch_engine(engine)
        client = TestClient(srv.app)
        try:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "weather?"}],
                    "tools": SAMPLE_TOOLS,
                    "tool_choice": "required",
                    "max_tokens": 64,
                },
            )
        finally:
            _restore_engine(orig_engine, orig_model)

        assert response.status_code == 200
        # Verify system message was injected
        sys_msgs = [
            m for m in engine.captured_messages if m.get("role") == "system"
        ]
        assert any("MUST call" in m["content"] for m in sys_msgs)

    def test_tool_choice_named_filters_tools(self):
        engine = FakeEngine(text=TOOL_CALL_MARKUP)
        orig_engine, orig_model = _patch_engine(engine)
        client = TestClient(srv.app)
        try:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "weather?"}],
                    "tools": SAMPLE_TOOLS,
                    "tool_choice": {"function": {"name": "get_weather"}},
                    "max_tokens": 64,
                },
            )
        finally:
            _restore_engine(orig_engine, orig_model)

        assert response.status_code == 200
        # Verify tools were filtered to only get_weather
        template_tools = engine.captured_kwargs.get("tools", [])
        assert len(template_tools) == 1
        assert template_tools[0]["function"]["name"] == "get_weather"
        # Verify system message mentions the function
        sys_msgs = [
            m for m in engine.captured_messages if m.get("role") == "system"
        ]
        assert any("get_weather" in m["content"] for m in sys_msgs)

    def test_tool_choice_auto_no_changes(self):
        engine = FakeEngine(text="Just plain text, no tools.")
        orig_engine, orig_model = _patch_engine(engine)
        client = TestClient(srv.app)
        try:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": SAMPLE_TOOLS,
                    "tool_choice": "auto",
                    "max_tokens": 64,
                },
            )
        finally:
            _restore_engine(orig_engine, orig_model)

        assert response.status_code == 200
        # tools should still be in kwargs
        assert "tools" in engine.captured_kwargs
        assert len(engine.captured_kwargs["tools"]) == 2
        # No extra system message injected
        sys_msgs = [
            m for m in engine.captured_messages if m.get("role") == "system"
        ]
        assert not any("MUST call" in m.get("content", "") for m in sys_msgs)

    def test_tool_choice_omitted_behaves_as_auto(self):
        engine = FakeEngine(text="Plain text response.")
        orig_engine, orig_model = _patch_engine(engine)
        client = TestClient(srv.app)
        try:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": SAMPLE_TOOLS,
                    "max_tokens": 64,
                },
            )
        finally:
            _restore_engine(orig_engine, orig_model)

        assert response.status_code == 200
        assert "tools" in engine.captured_kwargs
        assert len(engine.captured_kwargs["tools"]) == 2
