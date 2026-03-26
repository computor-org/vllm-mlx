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


# ---------------------------------------------------------------------------
# Lazy grammar triggers for tool_choice=auto
# ---------------------------------------------------------------------------

_INACTIVE = 0
_PENDING = 1
_ACTIVE = 2


class LazyToolCallProcessor:
    """Logits processor that activates grammar constraints lazily.

    Stays inactive (passes logits through) until the model emits a trigger
    token (e.g., ``<tool_call>``). Then activates the inner grammar-constrained
    processor for the JSON body. Deactivates on the end token (e.g.,
    ``</tool_call>``) and re-arms for the next tool call.

    Shape contract: mlx_lm's ``generate_step`` passes 1D ``input_ids``
    ``(seq_len,)`` and 1D ``logits`` ``(vocab_size,)``. The inner Outlines
    processor normalizes to 2D internally via its own ``__call__``.
    """

    def __init__(
        self,
        inner: Any,
        trigger_tokens: frozenset[int],
        end_tokens: frozenset[int],
        prefix_skip: int = 0,
    ):
        self._inner = inner
        self._trigger_tokens = trigger_tokens
        self._end_tokens = end_tokens
        self._prefix_skip = prefix_skip
        self._state = _INACTIVE
        self._prefix_remaining = 0
        self._activation_index = 0

    def __call__(self, input_ids: Any, logits: Any) -> Any:
        import mlx.core as mx

        last_token = input_ids[-1].item()

        if self._state == _INACTIVE:
            if last_token in self._trigger_tokens:
                self._state = _PENDING if self._prefix_skip > 0 else _ACTIVE
                self._prefix_remaining = self._prefix_skip
                self._inner.reset()
                self._activation_index = len(input_ids)
            return logits

        if self._state == _PENDING:
            self._prefix_remaining -= 1
            if self._prefix_remaining <= 0:
                self._state = _ACTIVE
                self._activation_index = len(input_ids)
            return logits

        # _ACTIVE
        if last_token in self._end_tokens:
            self._state = _INACTIVE
            return logits

        active_ids = input_ids[self._activation_index:]
        return self._inner(active_ids, logits)

    def reset(self) -> None:
        """Reset to inactive state for a new generation."""
        self._state = _INACTIVE
        self._prefix_remaining = 0
        self._activation_index = 0
        self._inner.reset()


def build_lazy_tool_call_processor(
    tools: list,
    trigger_tokens: frozenset[int],
    end_tokens: frozenset[int],
    prefix_skip: int = 0,
) -> Any:
    """Build a lazy tool call processor that activates on trigger tokens.

    Returns None when the Outlines backend is unavailable or tools are empty.
    """
    inner = build_tool_call_processor(tools)
    if inner is None:
        return None
    return LazyToolCallProcessor(inner, trigger_tokens, end_tokens, prefix_skip)
