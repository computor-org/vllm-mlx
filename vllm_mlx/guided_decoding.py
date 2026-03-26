"""Grammar-constrained decoding via Outlines.

Provides logits processors that constrain token generation to match a JSON
schema or regex pattern. Uses Outlines' OutlinesCoreBackend which compiles
grammars into efficient finite-state automata over the model's vocabulary.

The backend is initialized once after model load and reused for all requests.
When outlines is not installed, all functions degrade gracefully (returning
None) so callers can fall back to prompt-injection-based structured output.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_backend: Any = None


def init_guided_decoding(model: Any, tokenizer: Any) -> None:
    """Initialize the Outlines backend. Call once after model load."""
    global _backend
    try:
        from outlines.backends.outlines_core import OutlinesCoreBackend
        from outlines.models.mlxlm import MLXLM

        outlines_model = MLXLM(model, tokenizer)
        _backend = OutlinesCoreBackend(outlines_model)
        logger.info("Guided decoding initialized (outlines backend)")
    except ImportError:
        logger.info("outlines not installed; guided decoding disabled")
    except Exception:
        logger.warning("Failed to initialize guided decoding", exc_info=True)


def is_available() -> bool:
    """Return True when the Outlines backend has been initialized."""
    return _backend is not None


def build_json_schema_processor(schema: dict | str) -> Any:
    """Build a logits processor that constrains output to a JSON schema.

    Args:
        schema: JSON Schema as a dict or pre-serialized JSON string.

    Returns:
        A callable ``(input_ids, logits) -> logits`` suitable for mlx-lm's
        ``logits_processors`` parameter, or None when outlines is unavailable.
    """
    if _backend is None:
        return None
    schema_str = json.dumps(schema) if isinstance(schema, dict) else schema
    try:
        return _backend.get_json_schema_logits_processor(schema_str)
    except Exception:
        logger.warning("Failed to build JSON schema processor", exc_info=True)
        return None


def build_regex_processor(pattern: str) -> Any:
    """Build a logits processor that constrains output to a regex pattern.

    Args:
        pattern: Regular expression the generated text must match.

    Returns:
        A callable ``(input_ids, logits) -> logits`` suitable for mlx-lm's
        ``logits_processors`` parameter, or None when outlines is unavailable.
    """
    if _backend is None:
        return None
    try:
        return _backend.get_regex_logits_processor(pattern)
    except Exception:
        logger.warning("Failed to build regex processor", exc_info=True)
        return None


def build_tool_call_processor(tools: list) -> Any:
    """Build a logits processor for tool call JSON bodies.

    Creates a JSON schema matching ``{"name": "...", "arguments": {...}}``
    where *name* is constrained to declared tool names and *arguments*
    conform to each tool's parameter schema.  Returns None when outlines
    is unavailable or the tools list is empty.
    """
    if _backend is None or not tools:
        return None

    tool_schemas = []
    for t in tools:
        func = t.get("function", t) if isinstance(t, dict) else getattr(t, "function", None)
        if not func or not isinstance(func, dict):
            continue
        name = func.get("name")
        if not name:
            continue
        params = func.get("parameters", {"type": "object"})
        tool_schemas.append(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "const": name},
                    "arguments": params,
                },
                "required": ["name", "arguments"],
            }
        )

    if not tool_schemas:
        return None

    schema = tool_schemas[0] if len(tool_schemas) == 1 else {"anyOf": tool_schemas}
    return build_json_schema_processor(schema)
