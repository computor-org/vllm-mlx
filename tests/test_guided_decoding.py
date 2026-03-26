"""Tests for grammar-constrained decoding via Outlines."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx import guided_decoding


@pytest.fixture(autouse=True)
def _reset_backend():
    """Ensure each test starts with a clean backend state."""
    original = guided_decoding._backend
    guided_decoding._backend = None
    yield
    guided_decoding._backend = original


class TestIsAvailable:
    def test_false_when_not_initialized(self):
        assert guided_decoding.is_available() is False

    def test_true_when_backend_set(self):
        guided_decoding._backend = MagicMock()
        assert guided_decoding.is_available() is True


class TestInitGuidedDecoding:
    def test_graceful_when_outlines_missing(self):
        with patch.dict(
            sys.modules,
            {
                "outlines": None,
                "outlines.models": None,
                "outlines.models.mlxlm": None,
                "outlines.backends": None,
                "outlines.backends.outlines_core": None,
            },
        ):
            guided_decoding.init_guided_decoding(MagicMock(), MagicMock())
        assert guided_decoding._backend is None

    def test_graceful_on_backend_error(self):
        with patch(
            "vllm_mlx.guided_decoding.MLXLM",
            create=True,
            side_effect=RuntimeError("model mismatch"),
        ):
            guided_decoding.init_guided_decoding(MagicMock(), MagicMock())
        assert guided_decoding._backend is None


class TestBuildJsonSchemaProcessor:
    def test_returns_none_without_backend(self):
        result = guided_decoding.build_json_schema_processor({"type": "object"})
        assert result is None

    def test_with_dict_schema(self):
        mock_backend = MagicMock()
        mock_processor = MagicMock()
        mock_backend.get_json_schema_logits_processor.return_value = mock_processor
        guided_decoding._backend = mock_backend

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = guided_decoding.build_json_schema_processor(schema)

        assert result is mock_processor
        mock_backend.get_json_schema_logits_processor.assert_called_once_with(
            json.dumps(schema)
        )

    def test_with_string_schema(self):
        mock_backend = MagicMock()
        mock_backend.get_json_schema_logits_processor.return_value = MagicMock()
        guided_decoding._backend = mock_backend

        schema_str = '{"type": "object"}'
        guided_decoding.build_json_schema_processor(schema_str)

        mock_backend.get_json_schema_logits_processor.assert_called_once_with(
            schema_str
        )

    def test_returns_none_on_error(self):
        mock_backend = MagicMock()
        mock_backend.get_json_schema_logits_processor.side_effect = Exception("bad")
        guided_decoding._backend = mock_backend

        result = guided_decoding.build_json_schema_processor({"type": "invalid"})
        assert result is None


class TestBuildRegexProcessor:
    def test_returns_none_without_backend(self):
        result = guided_decoding.build_regex_processor(r"\d+")
        assert result is None

    def test_with_mock_backend(self):
        mock_backend = MagicMock()
        mock_processor = MagicMock()
        mock_backend.get_regex_logits_processor.return_value = mock_processor
        guided_decoding._backend = mock_backend

        result = guided_decoding.build_regex_processor(r"\d+")

        assert result is mock_processor
        mock_backend.get_regex_logits_processor.assert_called_once_with(r"\d+")

    def test_returns_none_on_error(self):
        mock_backend = MagicMock()
        mock_backend.get_regex_logits_processor.side_effect = Exception("bad regex")
        guided_decoding._backend = mock_backend

        result = guided_decoding.build_regex_processor(r"[invalid")
        assert result is None


class TestBuildGuidedProcessorHelper:
    """Tests for the server-side _build_guided_processor helper."""

    def test_returns_none_for_no_format(self):
        from vllm_mlx.server import _build_guided_processor

        assert _build_guided_processor(None) is None

    def test_returns_none_when_unavailable(self):
        from vllm_mlx.server import _build_guided_processor

        result = _build_guided_processor({"type": "json_schema", "json_schema": {"schema": {}}})
        # Backend not initialized, so should return None
        assert result is None

    def test_returns_none_for_text_format(self):
        from vllm_mlx.server import _build_guided_processor

        guided_decoding._backend = MagicMock()
        result = _build_guided_processor({"type": "text"})
        assert result is None

    def test_returns_none_for_json_object_format(self):
        from vllm_mlx.server import _build_guided_processor

        guided_decoding._backend = MagicMock()
        result = _build_guided_processor({"type": "json_object"})
        assert result is None

    def test_delegates_to_build_json_schema_processor(self):
        from vllm_mlx.server import _build_guided_processor

        mock_processor = MagicMock()
        mock_backend = MagicMock()
        mock_backend.get_json_schema_logits_processor.return_value = mock_processor
        guided_decoding._backend = mock_backend

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        rf = {
            "type": "json_schema",
            "json_schema": {"name": "test", "schema": schema},
        }
        result = _build_guided_processor(rf)

        assert result is mock_processor
        mock_backend.get_json_schema_logits_processor.assert_called_once_with(
            json.dumps(schema)
        )

    def test_handles_pydantic_response_format(self):
        from vllm_mlx.api.models import ResponseFormat, ResponseFormatJsonSchema
        from vllm_mlx.server import _build_guided_processor

        mock_processor = MagicMock()
        mock_backend = MagicMock()
        mock_backend.get_json_schema_logits_processor.return_value = mock_processor
        guided_decoding._backend = mock_backend

        schema = {"type": "object"}
        rf = ResponseFormat(
            type="json_schema",
            json_schema=ResponseFormatJsonSchema(
                name="test_schema",
                **{"schema": schema},
            ),
        )
        result = _build_guided_processor(rf)
        assert result is mock_processor


# ---------------------------------------------------------------------------
# Tests for build_tool_call_processor
# ---------------------------------------------------------------------------


class TestBuildToolCallProcessor:
    """Unit tests for build_tool_call_processor()."""

    def test_returns_none_without_backend(self):
        guided_decoding._backend = None
        result = guided_decoding.build_tool_call_processor(
            [{"function": {"name": "f", "parameters": {"type": "object"}}}]
        )
        assert result is None

    def test_returns_none_for_empty_tools(self):
        guided_decoding._backend = MagicMock()
        result = guided_decoding.build_tool_call_processor([])
        assert result is None
        guided_decoding._backend = None

    def test_single_tool_schema(self):
        mock_backend = MagicMock()
        mock_processor = MagicMock()
        mock_backend.get_json_schema_logits_processor.return_value = mock_processor
        guided_decoding._backend = mock_backend

        tools = [
            {
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            }
        ]
        result = guided_decoding.build_tool_call_processor(tools)
        assert result is mock_processor

        # Verify the schema passed to the backend
        call_args = mock_backend.get_json_schema_logits_processor.call_args[0][0]
        schema = json.loads(call_args)
        assert schema["properties"]["name"]["const"] == "get_weather"
        assert schema["properties"]["arguments"]["properties"]["city"]["type"] == "string"
        assert schema["required"] == ["name", "arguments"]
        guided_decoding._backend = None

    def test_multiple_tools_anyof(self):
        mock_backend = MagicMock()
        mock_backend.get_json_schema_logits_processor.return_value = MagicMock()
        guided_decoding._backend = mock_backend

        tools = [
            {"function": {"name": "get_weather", "parameters": {"type": "object"}}},
            {"function": {"name": "search", "parameters": {"type": "object"}}},
        ]
        result = guided_decoding.build_tool_call_processor(tools)
        assert result is not None

        call_args = mock_backend.get_json_schema_logits_processor.call_args[0][0]
        schema = json.loads(call_args)
        assert "anyOf" in schema
        assert len(schema["anyOf"]) == 2
        names = [s["properties"]["name"]["const"] for s in schema["anyOf"]]
        assert "get_weather" in names
        assert "search" in names
        guided_decoding._backend = None

    def test_tool_without_parameters(self):
        mock_backend = MagicMock()
        mock_backend.get_json_schema_logits_processor.return_value = MagicMock()
        guided_decoding._backend = mock_backend

        tools = [{"function": {"name": "ping"}}]
        result = guided_decoding.build_tool_call_processor(tools)
        assert result is not None

        call_args = mock_backend.get_json_schema_logits_processor.call_args[0][0]
        schema = json.loads(call_args)
        assert schema["properties"]["arguments"]["type"] == "object"
        guided_decoding._backend = None

    def test_malformed_tool_skipped(self):
        mock_backend = MagicMock()
        mock_backend.get_json_schema_logits_processor.return_value = MagicMock()
        guided_decoding._backend = mock_backend

        tools = [
            {"function": {"description": "no name"}},  # malformed
            {"function": {"name": "valid"}},
        ]
        result = guided_decoding.build_tool_call_processor(tools)
        assert result is not None

        call_args = mock_backend.get_json_schema_logits_processor.call_args[0][0]
        schema = json.loads(call_args)
        assert schema["properties"]["name"]["const"] == "valid"
        guided_decoding._backend = None
