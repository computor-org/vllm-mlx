# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for GPT-OSS models using Harmony format.

Harmony uses channels for reasoning vs final content:

    <|channel|>analysis
    <|message|>Let me think about this...
    <|end|>
    <|channel|>final
    <|message|>The answer is 42.
    <|return|>

The analysis channel contains reasoning, and the final channel
contains the user-facing response.
"""

import re

from .base import DeltaMessage, ReasoningParser

# Analysis channel blocks: <|channel|>analysis<|message|>...<|end|>
_ANALYSIS_PATTERN = re.compile(
    r"<\|channel\|>analysis\s*<\|message\|>(.*?)<\|end\|>",
    re.DOTALL,
)

# Final channel content: <|channel|>final<|message|>...<|return|>
_FINAL_PATTERN = re.compile(
    r"<\|channel\|>final\s*<\|message\|>(.*?)<\|return\|>",
    re.DOTALL,
)


class HarmonyReasoningParser(ReasoningParser):
    """
    Reasoning parser for GPT-OSS models using Harmony format.

    Extracts reasoning from the 'analysis' channel and content from
    the 'final' channel. Commentary channels (tool calls) are ignored
    since they are handled by the tool parser.

    Example:
        Input: "<|channel|>analysis<|message|>Thinking...<|end|>
                <|channel|>final<|message|>Result.<|return|>"
        Output: reasoning="Thinking...", content="Result."
    """

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._current_channel: str | None = None
        self._in_message: bool = False
        self._pending_channel_text = ""
        self._collecting_channel = False

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from complete Harmony output.

        Collects all analysis channel blocks as reasoning and the
        final channel block as content.

        Args:
            model_output: Complete model output text.

        Returns:
            (reasoning, content) tuple. Either may be None.
        """
        # Collect all analysis blocks
        analysis_blocks = _ANALYSIS_PATTERN.findall(model_output)
        reasoning = "\n".join(block.strip() for block in analysis_blocks) or None

        # Extract final channel content
        final_match = _FINAL_PATTERN.search(model_output)
        content = final_match.group(1).strip() if final_match else None

        return reasoning, content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from streaming Harmony output.

        Tracks the current channel and emits reasoning deltas for
        analysis channel content and content deltas for final channel.

        Args:
            previous_text: Accumulated text before this delta.
            current_text: Accumulated text including this delta.
            delta_text: The new text in this streaming chunk.

        Returns:
            DeltaMessage with reasoning and/or content, or None.
        """
        if self._collecting_channel:
            self._pending_channel_text += delta_text
            resolved = self._try_resolve_pending_channel()
            if resolved is not None:
                return resolved

        # Detect channel switches in the delta
        if "<|channel|>" in delta_text:
            self._collecting_channel = True
            self._pending_channel_text = delta_text
            resolved = self._try_resolve_pending_channel()
            if resolved is not None:
                return resolved
            return None

        # Detect channel from full context if not yet determined
        if self._current_channel is None and "<|channel|>" in current_text:
            last_channel = current_text.rfind("<|channel|>")
            after = current_text[last_channel + len("<|channel|>") :]
            self._current_channel = self._resolve_channel_name(after)

        # Commentary is routed through the tool parser via DeltaMessage.content.
        if self._current_channel == "commentary":
            if "<|message|>" in delta_text:
                self._in_message = True
            if any(token in delta_text for token in ("<|call|>", "<|end|>", "<|return|>")):
                self._in_message = False
            return DeltaMessage(content=delta_text)

        # Handle message start
        if "<|message|>" in delta_text:
            self._in_message = True
            # Don't emit the token itself
            return None

        # Handle channel/message end tokens
        if any(
            token in delta_text
            for token in ("<|end|>", "<|return|>", "<|call|>", "<|start|>")
        ):
            self._in_message = False
            return None

        # Skip control tokens
        if delta_text.strip().startswith("<|") and delta_text.strip().endswith("|>"):
            return None

        # Emit content based on current channel
        if self._in_message and self._current_channel == "analysis":
            return DeltaMessage(reasoning=delta_text)

        if self._in_message and self._current_channel == "final":
            return DeltaMessage(content=delta_text)

        # In commentary or unknown channel, suppress
        return None

    def reset_state(self):
        """Reset streaming state for a new request."""
        self._current_channel = None
        self._in_message = False
        self._pending_channel_text = ""
        self._collecting_channel = False

    @staticmethod
    def _resolve_channel_name(text: str) -> str | None:
        """Extract the active Harmony channel name from a delta or suffix."""
        if "analysis" in text:
            return "analysis"
        if "final" in text:
            return "final"
        if "commentary" in text:
            return "commentary"
        return None

    def _try_resolve_pending_channel(self) -> DeltaMessage | None:
        """Resolve a partially streamed channel header once enough text arrives."""
        suffix = self._pending_channel_text.split("<|channel|>", 1)[-1]
        channel_name = self._resolve_channel_name(suffix)
        if channel_name is None:
            return None

        buffered = self._pending_channel_text
        self._pending_channel_text = ""
        self._collecting_channel = False
        self._current_channel = channel_name
        self._in_message = "<|message|>" in buffered

        if channel_name == "commentary":
            return DeltaMessage(content=buffered)

        if "<|message|>" in buffered:
            content = buffered.split("<|message|>", 1)[1]
            if content:
                if channel_name == "analysis":
                    return DeltaMessage(reasoning=content)
                return DeltaMessage(content=content)
        return None
