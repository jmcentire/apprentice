"""
PII Detection Strategies — multi-modal detection pipeline.

Provides a unified detection framework that combines:
  - RegexDetectionStrategy: fast pattern-based detection (emails, phones, SSNs, etc.)
  - FieldHeuristicStrategy: field-path-based detection for sensitive fields
  - NERDetectionStrategy: ML-based named entity recognition (optional, lazy-loaded)

Detection is separated from replacement so multiple strategies can contribute
before any tokenization occurs.
"""

import logging
import re
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

_PACT_KEY = "PACT:pii_detection"
logger = logging.getLogger(__name__)


def _log(level: str, msg: str, **kwargs: Any) -> None:
    getattr(logger, level)(f"[{_PACT_KEY}] {msg}", **kwargs)


# ===========================================================================
# Detection span model
# ===========================================================================


class PIIDetection(BaseModel):
    """A single detected PII span within a text string.

    Attributes:
        start: Character offset (inclusive).
        end: Character offset (exclusive).
        value: The matched text.
        entity_type: Category of PII (e.g. "email", "person_name").
        confidence: Detection confidence in [0.0, 1.0].
        source: Which strategy produced this detection.
    """
    model_config = ConfigDict(frozen=True)

    start: int
    end: int
    value: str
    entity_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str


# ===========================================================================
# Detection strategy protocol
# ===========================================================================


@runtime_checkable
class DetectionStrategy(Protocol):
    """Protocol for PII detection strategies."""

    def detect(self, text: str, field_path: str = "") -> list[PIIDetection]: ...

    @property
    def name(self) -> str: ...


# ===========================================================================
# Built-in patterns (shared with pii_tokenizer for backward compat)
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
# Regex Detection Strategy
# ===========================================================================


class RegexDetectionStrategy:
    """Detects PII using compiled regular expressions.

    Extracts from the existing ``_build_patterns`` + ``_scan_and_replace`` logic
    but produces ``PIIDetection`` spans instead of inline replacement.
    SSN-before-phone ordering is preserved to avoid partial matches.
    """

    def __init__(
        self,
        enabled_entity_types: list[str] | None = None,
        custom_patterns: dict[str, str] | None = None,
    ) -> None:
        if enabled_entity_types is None:
            enabled_entity_types = list(_BUILTIN_PATTERNS.keys())
        self._compiled = self._build(enabled_entity_types, custom_patterns or {})

    @property
    def name(self) -> str:
        return "regex"

    @staticmethod
    def _build(
        enabled: list[str], custom: dict[str, str]
    ) -> list[tuple[str, re.Pattern[str]]]:
        patterns: list[tuple[str, re.Pattern[str]]] = []
        # SSN must precede phone to avoid partial matches
        ordered: list[str] = []
        if "ssn" in enabled:
            ordered.append("ssn")
        for et in enabled:
            if et != "ssn" and et in _BUILTIN_PATTERNS:
                ordered.append(et)
        for entity_type in ordered:
            patterns.append((entity_type, re.compile(_BUILTIN_PATTERNS[entity_type])))
        for name, raw in custom.items():
            patterns.append((name, re.compile(raw)))
        return patterns

    def detect(self, text: str, field_path: str = "") -> list[PIIDetection]:
        detections: list[PIIDetection] = []
        for entity_type, pattern in self._compiled:
            for m in pattern.finditer(text):
                detections.append(
                    PIIDetection(
                        start=m.start(),
                        end=m.end(),
                        value=m.group(),
                        entity_type=entity_type,
                        confidence=1.0,
                        source="regex",
                    )
                )
        return detections


# ===========================================================================
# Field Heuristic Strategy
# ===========================================================================


class FieldHeuristicStrategy:
    """Detects PII by matching field paths against known sensitive-field names.

    When the field path matches a sensitive field, the entire text value is
    treated as PII with the entity type derived from the field name.
    """

    # Default set of field names that commonly contain PII
    DEFAULT_SENSITIVE_FIELDS: frozenset[str] = frozenset({
        "name", "first_name", "last_name", "full_name",
        "email", "email_address",
        "phone", "phone_number", "mobile", "telephone",
        "ssn", "social_security_number",
        "address", "street_address", "mailing_address",
        "date_of_birth", "dob", "birthday",
        "credit_card", "card_number",
        "passport", "passport_number",
        "driver_license", "license_number",
        "password", "secret", "api_key",
    })

    def __init__(
        self,
        sensitive_fields: set[str] | None = None,
        learned_fields: set[str] | None = None,
    ) -> None:
        base = set(self.DEFAULT_SENSITIVE_FIELDS) if sensitive_fields is None else set(sensitive_fields)
        if learned_fields:
            base |= learned_fields
        self._sensitive: frozenset[str] = frozenset(base)

    @property
    def name(self) -> str:
        return "field_heuristic"

    def detect(self, text: str, field_path: str = "") -> list[PIIDetection]:
        if not field_path or not text:
            return []
        # Check the leaf field name against sensitive fields
        leaf = field_path.rsplit(".", 1)[-1].lower()
        if leaf not in self._sensitive:
            return []
        # Map common field names to entity types
        entity_type = self._field_to_entity_type(leaf)
        return [
            PIIDetection(
                start=0,
                end=len(text),
                value=text,
                entity_type=entity_type,
                confidence=0.9,
                source="field_heuristic",
            )
        ]

    @staticmethod
    def _field_to_entity_type(field_name: str) -> str:
        mapping: dict[str, str] = {
            "name": "person_name",
            "first_name": "person_name",
            "last_name": "person_name",
            "full_name": "person_name",
            "email": "email",
            "email_address": "email",
            "phone": "phone",
            "phone_number": "phone",
            "mobile": "phone",
            "telephone": "phone",
            "ssn": "ssn",
            "social_security_number": "ssn",
            "address": "address",
            "street_address": "address",
            "mailing_address": "address",
            "date_of_birth": "date_of_birth",
            "dob": "date_of_birth",
            "birthday": "date_of_birth",
            "credit_card": "credit_card",
            "card_number": "credit_card",
            "passport": "passport",
            "passport_number": "passport",
            "driver_license": "driver_license",
            "license_number": "driver_license",
            "password": "sensitive_field",
            "secret": "sensitive_field",
            "api_key": "sensitive_field",
        }
        return mapping.get(field_name, "sensitive_field")


# ===========================================================================
# NER Detection Strategy (optional ML dependency)
# ===========================================================================


# NER label -> our entity types
_NER_LABEL_MAP: dict[str, str] = {
    "PER": "person_name",
    "PERSON": "person_name",
    "LOC": "location",
    "LOCATION": "location",
    "GPE": "location",
    "ORG": "organization",
    "ORGANIZATION": "organization",
    "MISC": "misc_entity",
    "DATE": "date",
    "NORP": "group",
    "FAC": "facility",
    "PRODUCT": "product",
    "EVENT": "event",
}


class NERDetectionStrategy:
    """Detects PII using a HuggingFace token-classification (NER) pipeline.

    Lazy-loads ``transformers`` on first call. If ``transformers`` or ``torch``
    are not installed, disables itself gracefully (same pattern as
    ``SemanticSimilarityEvaluator``).
    """

    def __init__(
        self,
        model_name: str = "dslim/bert-base-NER",
        device: str = "cpu",
        confidence_threshold: float = 0.7,
        max_text_length: int = 10000,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._max_text_length = max_text_length
        self._pipeline: Any = None
        self._available: bool | None = None  # None = not yet checked

    @property
    def name(self) -> str:
        return "ner_model"

    def _ensure_loaded(self) -> bool:
        """Lazy-load the NER pipeline. Returns True if available."""
        if self._available is not None:
            return self._available
        try:
            from transformers import pipeline as hf_pipeline
            self._pipeline = hf_pipeline(
                "token-classification",
                model=self._model_name,
                device=self._device,
                aggregation_strategy="simple",
            )
            self._available = True
            _log("info", f"NER model loaded: {self._model_name}")
        except ImportError:
            _log("warning", "transformers/torch not installed — NER strategy disabled")
            self._available = False
        except Exception as exc:
            _log("warning", f"Failed to load NER model '{self._model_name}': {exc}")
            self._available = False
        return self._available

    def detect(self, text: str, field_path: str = "") -> list[PIIDetection]:
        if not self._ensure_loaded():
            return []
        if not text or len(text) > self._max_text_length:
            return []

        try:
            results = self._pipeline(text)
        except Exception as exc:
            _log("warning", f"NER inference error: {exc}")
            return []

        detections: list[PIIDetection] = []
        for ent in results:
            score = float(ent.get("score", 0.0))
            if score < self._confidence_threshold:
                continue
            label = ent.get("entity_group", ent.get("entity", "MISC"))
            entity_type = _NER_LABEL_MAP.get(label, label.lower())
            word = ent.get("word", "")
            start = int(ent.get("start", 0))
            end = int(ent.get("end", start + len(word)))
            # Use actual text slice if offsets are available
            value = text[start:end] if start < end <= len(text) else word
            detections.append(
                PIIDetection(
                    start=start,
                    end=end,
                    value=value,
                    entity_type=entity_type,
                    confidence=score,
                    source="ner_model",
                )
            )
        return detections


# ===========================================================================
# Composite Detection Pipeline
# ===========================================================================


def _merge_detections(detections: list[PIIDetection]) -> list[PIIDetection]:
    """Merge overlapping detection spans, keeping the highest confidence.

    Sort by start position, then resolve overlaps: when two spans overlap,
    keep the one with higher confidence (union strategy — recall over precision).
    """
    if not detections:
        return []
    # Sort by start, then by descending confidence for tie-breaking
    sorted_dets = sorted(detections, key=lambda d: (d.start, -d.confidence))
    merged: list[PIIDetection] = [sorted_dets[0]]
    for det in sorted_dets[1:]:
        last = merged[-1]
        if det.start < last.end:
            # Overlap — keep the one with higher confidence
            if det.confidence > last.confidence:
                merged[-1] = det
            # else keep last (already has higher or equal confidence)
        else:
            merged.append(det)
    return merged


class CompositeDetectionPipeline:
    """Runs multiple detection strategies and merges results.

    Error-resilient: if any strategy raises, it is logged and skipped.
    """

    def __init__(self, strategies: list[Any] | None = None) -> None:
        self._strategies: list[Any] = strategies or []

    def add_strategy(self, strategy: Any) -> None:
        self._strategies.append(strategy)

    @property
    def strategies(self) -> list[Any]:
        return list(self._strategies)

    def detect_all(self, text: str, field_path: str = "") -> list[PIIDetection]:
        """Run all strategies and return merged, deduplicated detections."""
        all_detections: list[PIIDetection] = []
        for strategy in self._strategies:
            try:
                dets = strategy.detect(text, field_path)
                all_detections.extend(dets)
            except Exception as exc:
                strategy_name = getattr(strategy, "name", type(strategy).__name__)
                _log("warning", f"Strategy '{strategy_name}' failed: {exc}")
        return _merge_detections(all_detections)
