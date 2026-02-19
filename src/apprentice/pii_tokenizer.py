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
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from apprentice.middleware import MiddlewareContext, MiddlewareResponse


logger = logging.getLogger(__name__)


# ===========================================================================
# Built-in PII patterns
# ===========================================================================

_BUILTIN_PATTERNS: dict[str, str] = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "api_key": r"\b(?:sk|pk|key|token|api)[-_][A-Za-z0-9]{20,}\b",
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
        sensitive_fields: Field paths that are always tokenized regardless of content.
        learned_patterns_path: Path to the learned patterns YAML file.
        detection_mode: Detection strategy mode — regex_only, hybrid, or ner_only.
        ner_model_path: HuggingFace model ID for NER (used when detection_mode != regex_only).
        ner_device: Device for NER inference (cpu or cuda).
        ner_confidence_threshold: Minimum confidence for NER detections.
        ner_max_text_length: Skip NER on texts longer than this.
    """
    model_config = ConfigDict(frozen=True)

    enabled_entity_types: list[str] = Field(
        default_factory=lambda: ["email", "phone", "ssn", "credit_card", "ip_address", "api_key"]
    )
    custom_patterns: dict[str, str] = Field(default_factory=dict)
    token_format: str = "<PII:{type}:{hash}>"
    sensitive_fields: list[str] = Field(default_factory=lambda: [
        "password", "secret", "api_key"
    ])
    learned_patterns_path: str | None = None
    detection_mode: str = "regex_only"
    ner_model_path: str | None = None
    ner_device: str = "cpu"
    ner_confidence_threshold: float = 0.7
    ner_max_text_length: int = 10000


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
# Learned Pattern Store
# ===========================================================================


class FieldPIIStats(BaseModel):
    """Statistics for a single (field_path, entity_type) pair."""
    model_config = ConfigDict(frozen=False)

    field_path: str
    entity_type: str
    confidence: str = "low"  # low, medium, high
    first_seen: str = ""
    match_count: int = 0
    detection_sources: dict[str, int] = Field(default_factory=dict)
    false_positive_count: int = 0
    last_seen: str = ""

    @property
    def computed_confidence(self) -> float:
        """Numeric confidence based on match count, multi-source corroboration, and FP penalty."""
        if self.match_count == 0:
            return 0.0
        base = min(self.match_count / 10.0, 1.0)
        source_count = len(self.detection_sources)
        source_bonus = min(source_count * 0.05, 0.1) if source_count > 1 else 0.0
        fp_penalty = min(self.false_positive_count * 0.1, 0.5)
        return max(0.0, min(1.0, base + source_bonus - fp_penalty))


class LearnedPatternStore:
    """Persists learned PII field patterns to a YAML file.

    Accumulates which field paths consistently contain PII matches and
    flushes to disk periodically. On startup, loads existing learnings.

    The file is intended to be .gitignored and user-editable.
    """

    # Flush every N record_match calls
    _FLUSH_INTERVAL = 50

    def __init__(self, path: str = ".apprentice/pii_learned_patterns.yaml") -> None:
        self._path = Path(path)
        self._field_stats: dict[tuple[str, str], FieldPIIStats] = {}
        self._calls_since_flush = 0
        self._load()

    def record_match(self, field_path: str, entity_type: str, source: str = "regex") -> None:
        """Record that a PII match was found at this field path."""
        key = (field_path, entity_type)
        now = datetime.now(timezone.utc).isoformat()
        if key not in self._field_stats:
            self._field_stats[key] = FieldPIIStats(
                field_path=field_path,
                entity_type=entity_type,
                first_seen=now,
            )
        stats = self._field_stats[key]
        stats.match_count += 1
        stats.last_seen = now
        stats.detection_sources[source] = stats.detection_sources.get(source, 0) + 1
        # Update confidence based on match count
        if stats.match_count >= 20:
            stats.confidence = "high"
        elif stats.match_count >= 5:
            stats.confidence = "medium"

        self._calls_since_flush += 1
        if self._calls_since_flush >= self._FLUSH_INTERVAL:
            self.flush()

    def record_false_positive(self, field_path: str, entity_type: str) -> None:
        """Record that a detection at this field was a false positive."""
        key = (field_path, entity_type)
        if key in self._field_stats:
            self._field_stats[key].false_positive_count += 1
            self.flush()

    def get_known_fields(self) -> list[tuple[str, str]]:
        """Return (field_path, entity_type) pairs from learned data."""
        return list(self._field_stats.keys())

    def get_high_confidence_fields(self) -> list[tuple[str, str]]:
        """Return (field_path, entity_type) pairs with high confidence."""
        return [
            (stats.field_path, stats.entity_type)
            for stats in self._field_stats.values()
            if stats.confidence == "high"
        ]

    def flush(self) -> None:
        """Write accumulated learnings to disk."""
        self._calls_since_flush = 0
        if not self._field_stats:
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)

        learned_fields = []
        for stats in sorted(self._field_stats.values(), key=lambda s: (-s.match_count, s.field_path)):
            learned_fields.append({
                "field_path": stats.field_path,
                "entity_type": stats.entity_type,
                "confidence": stats.confidence,
                "first_seen": stats.first_seen,
                "match_count": stats.match_count,
                "detection_sources": stats.detection_sources,
                "false_positive_count": stats.false_positive_count,
                "last_seen": stats.last_seen,
            })

        data = {"learned_fields": learned_fields}
        self._path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

    def _load(self) -> None:
        """Load existing learnings from disk if file exists."""
        if not self._path.exists():
            return
        try:
            raw = yaml.safe_load(self._path.read_text())
            if not raw or "learned_fields" not in raw:
                return
            for entry in raw["learned_fields"]:
                key = (entry["field_path"], entry["entity_type"])
                self._field_stats[key] = FieldPIIStats(
                    field_path=entry["field_path"],
                    entity_type=entry["entity_type"],
                    confidence=entry.get("confidence", "low"),
                    first_seen=entry.get("first_seen", ""),
                    match_count=entry.get("match_count", 0),
                    detection_sources=entry.get("detection_sources", {}),
                    false_positive_count=entry.get("false_positive_count", 0),
                    last_seen=entry.get("last_seen", ""),
                )
        except Exception:
            logger.warning("Failed to load learned patterns from %s", self._path)


# ===========================================================================
# Recursive scan / replace helpers
# ===========================================================================


def _scan_and_replace(
    data: Any,
    compiled_patterns: list[tuple[str, re.Pattern]],
    registry: TokenRegistry,
    sensitive_field_names: set[str] | None = None,
    _current_path: str = "",
    _learned_store: "LearnedPatternStore | None" = None,
) -> Any:
    """Recursively walk *data* and replace PII matches with tokens.

    If *sensitive_field_names* is provided and the current field name matches,
    the entire string value is tokenized as entity type ``"sensitive_field"``.
    """
    if isinstance(data, str):
        # Check if current field path is a sensitive field
        if sensitive_field_names and _current_path:
            # Match against the leaf field name or the full dot-path
            leaf = _current_path.rsplit(".", 1)[-1] if "." in _current_path else _current_path
            if leaf in sensitive_field_names or _current_path in sensitive_field_names:
                token = registry.tokenize(data, "sensitive_field")
                if _learned_store:
                    _learned_store.record_match(_current_path, "sensitive_field")
                return token

        result = data
        for entity_type, pattern in compiled_patterns:
            def _replace(m: re.Match, et: str = entity_type) -> str:
                token = registry.tokenize(m.group(), et)
                if _learned_store and _current_path:
                    _learned_store.record_match(_current_path, et)
                return token
            result = pattern.sub(_replace, result)
        return result

    if isinstance(data, dict):
        return {
            k: _scan_and_replace(
                v, compiled_patterns, registry, sensitive_field_names,
                f"{_current_path}.{k}" if _current_path else k,
                _learned_store,
            )
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [
            _scan_and_replace(
                item, compiled_patterns, registry, sensitive_field_names,
                _current_path, _learned_store,
            )
            for item in data
        ]
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
# Detect-then-replace helpers (new two-phase approach)
# ===========================================================================


def _replace_detections(
    text: str,
    detections: list,
    registry: TokenRegistry,
) -> str:
    """Replace detected PII spans with tokens, processing from right to left."""
    if not detections:
        return text
    # Sort by start position descending so replacements don't shift offsets
    sorted_dets = sorted(detections, key=lambda d: d.start, reverse=True)
    result = text
    for det in sorted_dets:
        token = registry.tokenize(det.value, det.entity_type)
        result = result[:det.start] + token + result[det.end:]
    return result


def _detect_and_replace(
    data: Any,
    pipeline: Any,
    registry: TokenRegistry,
    learned_store: "LearnedPatternStore | None" = None,
    _current_path: str = "",
) -> Any:
    """Recursively walk *data*, detect PII using the composite pipeline,
    and replace detected spans with tokens.

    This is the two-phase replacement: detect all spans first, then replace.
    """
    if isinstance(data, str):
        detections = pipeline.detect_all(data, field_path=_current_path)
        if learned_store and detections:
            for det in detections:
                learned_store.record_match(_current_path, det.entity_type, det.source)
        return _replace_detections(data, detections, registry)
    if isinstance(data, dict):
        return {
            k: _detect_and_replace(
                v, pipeline, registry, learned_store,
                f"{_current_path}.{k}" if _current_path else k,
            )
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [
            _detect_and_replace(
                item, pipeline, registry, learned_store,
                f"{_current_path}[]",
            )
            for item in data
        ]
    return data


# ===========================================================================
# PIITokenizer — Middleware implementation
# ===========================================================================


class PIITokenizer:
    """Middleware that tokenizes PII in ``input_data`` before the model sees it,
    and restores original values in ``output_data`` after the model responds.

    Implements the ``Middleware`` protocol from ``apprentice.middleware``.

    When a ``detection_pipeline`` is provided, uses the two-phase
    detect-then-replace approach. Otherwise falls back to the legacy
    regex-only ``_scan_and_replace``.
    """

    def __init__(
        self,
        config: PIITokenizerConfig | None = None,
        learned_store: LearnedPatternStore | None = None,
        detection_pipeline: Any | None = None,
    ) -> None:
        self._config = config or PIITokenizerConfig()
        self._compiled_patterns = self._build_patterns()
        self._learned_store = learned_store
        self._detection_pipeline = detection_pipeline
        self._sensitive_field_names: set[str] = set(self._config.sensitive_fields)

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

        When a detection pipeline is available, uses the two-phase
        detect-then-replace approach for multi-modal detection.
        """
        registry = TokenRegistry(self._config.token_format)

        if self._detection_pipeline is not None:
            tokenized_input = _detect_and_replace(
                context.input_data,
                self._detection_pipeline,
                registry,
                self._learned_store,
            )
        else:
            tokenized_input = _scan_and_replace(
                context.input_data,
                self._compiled_patterns,
                registry,
                sensitive_field_names=self._sensitive_field_names,
                _learned_store=self._learned_store,
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
