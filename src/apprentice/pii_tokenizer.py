"""
PII Tokenizer Middleware — scans and replaces PII with opaque tokens.

Implements the Middleware protocol from apprentice.middleware.

Critical invariants:
  - Token registry is NEVER written to disk.
  - Training data store NEVER receives un-tokenized PII.
  - Model NEVER sees real PII values.
  - Tokens are opaque to the model.
"""

import hashlib
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from apprentice.middleware import MiddlewareContext, MiddlewareResponse


# ===========================================================================
# Built-in PII patterns
# ===========================================================================

_BUILTIN_PATTERNS: dict[str, str] = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
}


# ===========================================================================
# Configuration
# ===========================================================================


class PIITokenizerConfig(BaseModel):
    """Frozen configuration for the PII tokenizer middleware.

    Attributes:
        enabled_entity_types: Which built-in entity types to scan for.
        custom_patterns: Additional name-to-regex mappings beyond the built-ins.
        token_format: Python format string for replacement tokens.
            Must contain ``{type}`` and ``{hash}`` placeholders.
    """
    model_config = ConfigDict(frozen=True)

    enabled_entity_types: list[str] = Field(
        default_factory=lambda: ["email", "phone", "ssn", "credit_card"]
    )
    custom_patterns: dict[str, str] = Field(default_factory=dict)
    token_format: str = "<PII:{type}:{hash}>"


# ===========================================================================
# Token Registry (in-memory only — NEVER persisted)
# ===========================================================================


class TokenRegistry:
    """In-memory bidirectional mapping between PII values and opaque tokens.

    - Deterministic: the same (value, entity_type) pair always produces the
      same token within the lifetime of this registry instance.
    - The registry is NEVER serialized or written to disk.
    """

    def __init__(self, token_format: str = "<PII:{type}:{hash}>") -> None:
        self._token_format = token_format
        # Forward: token -> original value
        self._token_to_value: dict[str, str] = {}
        # Reverse: original value -> token (for determinism)
        self._value_to_token: dict[str, str] = {}

    def tokenize(self, value: str, entity_type: str) -> str:
        """Replace *value* with an opaque token.

        If the same *value* was already tokenized, returns the same token
        (deterministic within one registry instance).
        """
        if value in self._value_to_token:
            return self._value_to_token[value]

        short_hash = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
        token = self._token_format.format(type=entity_type, hash=short_hash)

        self._token_to_value[token] = value
        self._value_to_token[value] = token
        return token

    def detokenize(self, token: str) -> str:
        """Restore the original value from *token*.

        Returns the token unchanged if it is not recognized.
        """
        return self._token_to_value.get(token, token)

    def clear(self) -> None:
        """Wipe all mappings."""
        self._token_to_value.clear()
        self._value_to_token.clear()


# ===========================================================================
# Recursive scan / replace helpers
# ===========================================================================


def _scan_and_replace(
    data: Any,
    compiled_patterns: list[tuple[str, re.Pattern]],
    registry: TokenRegistry,
) -> Any:
    """Recursively walk *data* and replace PII matches with tokens."""
    if isinstance(data, str):
        result = data
        for entity_type, pattern in compiled_patterns:
            result = pattern.sub(
                lambda m, et=entity_type: registry.tokenize(m.group(), et),
                result,
            )
        return result
    if isinstance(data, dict):
        return {k: _scan_and_replace(v, compiled_patterns, registry) for k, v in data.items()}
    if isinstance(data, list):
        return [_scan_and_replace(item, compiled_patterns, registry) for item in data]
    return data


def _restore_tokens(data: Any, registry: TokenRegistry) -> Any:
    """Recursively walk *data* and restore original PII values from tokens."""
    if isinstance(data, str):
        result = data
        # Replace every token found in the string
        for token, original in registry._token_to_value.items():
            result = result.replace(token, original)
        return result
    if isinstance(data, dict):
        return {k: _restore_tokens(v, registry) for k, v in data.items()}
    if isinstance(data, list):
        return [_restore_tokens(item, registry) for item in data]
    return data


# ===========================================================================
# PIITokenizer — Middleware implementation
# ===========================================================================


class PIITokenizer:
    """Middleware that tokenizes PII in ``input_data`` before the model sees it,
    and restores original values in ``output_data`` after the model responds.

    Implements the ``Middleware`` protocol from ``apprentice.middleware``.
    """

    def __init__(self, config: PIITokenizerConfig | None = None) -> None:
        self._config = config or PIITokenizerConfig()
        self._compiled_patterns = self._build_patterns()

    def _build_patterns(self) -> list[tuple[str, re.Pattern]]:
        """Compile the enabled built-in patterns plus any custom patterns."""
        patterns: list[tuple[str, re.Pattern]] = []

        # SSN pattern must be checked before phone to avoid partial matches
        # when both are enabled. Order: ssn first, then everything else.
        ordered_types: list[str] = []
        if "ssn" in self._config.enabled_entity_types:
            ordered_types.append("ssn")
        for et in self._config.enabled_entity_types:
            if et != "ssn" and et in _BUILTIN_PATTERNS:
                ordered_types.append(et)

        for entity_type in ordered_types:
            raw = _BUILTIN_PATTERNS[entity_type]
            patterns.append((entity_type, re.compile(raw)))

        # Custom patterns (appended after built-ins)
        for name, raw in self._config.custom_patterns.items():
            patterns.append((name, re.compile(raw)))

        return patterns

    # ---- Middleware Protocol ------------------------------------------------

    def pre_process(self, context: MiddlewareContext) -> MiddlewareContext:
        """Scan ``input_data`` for PII and replace with opaque tokens.

        The ``TokenRegistry`` is stored in ``middleware_state["pii_token_registry"]``
        so that ``post_process`` can restore original values.
        """
        registry = TokenRegistry(self._config.token_format)
        tokenized_input = _scan_and_replace(
            context.input_data, self._compiled_patterns, registry
        )

        new_state = {**context.middleware_state, "pii_token_registry": registry}
        return context.model_copy(
            update={"input_data": tokenized_input, "middleware_state": new_state}
        )

    def post_process(
        self, context: MiddlewareContext, response: MiddlewareResponse
    ) -> MiddlewareResponse:
        """Restore original PII values in ``output_data`` from tokens.

        Reads the ``TokenRegistry`` from ``context.middleware_state["pii_token_registry"]``.
        If the registry is missing, returns the response unchanged.
        """
        registry: TokenRegistry | None = context.middleware_state.get(
            "pii_token_registry"
        )
        if registry is None:
            return response

        restored_output = _restore_tokens(response.output_data, registry)
        return response.model_copy(update={"output_data": restored_output})
