"""Tests for the PII detection strategies and composite pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from apprentice.pii_detection import (
    CompositeDetectionPipeline,
    FieldHeuristicStrategy,
    NERDetectionStrategy,
    PIIDetection,
    RegexDetectionStrategy,
    _merge_detections,
)


# ============================================================================
# PIIDetection model
# ============================================================================


class TestPIIDetection:
    def test_creation(self):
        d = PIIDetection(
            start=0, end=5, value="hello", entity_type="email",
            confidence=1.0, source="regex",
        )
        assert d.start == 0
        assert d.end == 5
        assert d.value == "hello"
        assert d.entity_type == "email"
        assert d.confidence == 1.0
        assert d.source == "regex"

    def test_frozen(self):
        d = PIIDetection(
            start=0, end=5, value="hello", entity_type="email",
            confidence=1.0, source="regex",
        )
        with pytest.raises(Exception):
            d.start = 10

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            PIIDetection(
                start=0, end=1, value="x", entity_type="t",
                confidence=1.5, source="s",
            )
        with pytest.raises(Exception):
            PIIDetection(
                start=0, end=1, value="x", entity_type="t",
                confidence=-0.1, source="s",
            )


# ============================================================================
# RegexDetectionStrategy
# ============================================================================


class TestRegexDetectionStrategy:
    def test_name(self):
        s = RegexDetectionStrategy()
        assert s.name == "regex"

    def test_detect_email(self):
        s = RegexDetectionStrategy()
        dets = s.detect("Contact alice@example.com for info.")
        assert len(dets) == 1
        assert dets[0].value == "alice@example.com"
        assert dets[0].entity_type == "email"
        assert dets[0].confidence == 1.0
        assert dets[0].source == "regex"
        assert dets[0].start == 8
        assert dets[0].end == 25

    def test_detect_phone(self):
        s = RegexDetectionStrategy()
        dets = s.detect("Call 555-123-4567 today.")
        assert len(dets) == 1
        assert dets[0].value == "555-123-4567"
        assert dets[0].entity_type == "phone"

    def test_detect_phone_no_dashes(self):
        s = RegexDetectionStrategy()
        dets = s.detect("Call 5551234567 today.")
        assert len(dets) == 1
        assert dets[0].entity_type == "phone"

    def test_detect_ssn(self):
        s = RegexDetectionStrategy()
        dets = s.detect("SSN is 123-45-6789.")
        assert len(dets) == 1
        assert dets[0].value == "123-45-6789"
        assert dets[0].entity_type == "ssn"

    def test_ssn_before_phone_ordering(self):
        """SSN pattern must match before phone to avoid partial matches."""
        s = RegexDetectionStrategy()
        dets = s.detect("SSN: 123-45-6789")
        # Should detect as SSN, not phone
        ssn_dets = [d for d in dets if d.entity_type == "ssn"]
        assert len(ssn_dets) == 1
        assert ssn_dets[0].value == "123-45-6789"

    def test_detect_credit_card(self):
        s = RegexDetectionStrategy()
        dets = s.detect("Card: 4111-1111-1111-1111.")
        assert len(dets) == 1
        assert dets[0].entity_type == "credit_card"

    def test_detect_multiple(self):
        s = RegexDetectionStrategy()
        text = "Email alice@example.com or call 555-123-4567."
        dets = s.detect(text)
        types = {d.entity_type for d in dets}
        assert "email" in types
        assert "phone" in types

    def test_custom_patterns(self):
        s = RegexDetectionStrategy(
            enabled_entity_types=[],
            custom_patterns={"ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"},
        )
        dets = s.detect("Server at 192.168.1.1 is down.")
        assert len(dets) == 1
        assert dets[0].entity_type == "ip_address"
        assert dets[0].value == "192.168.1.1"

    def test_disabled_types(self):
        s = RegexDetectionStrategy(enabled_entity_types=["email"])
        dets = s.detect("Email alice@example.com, phone 555-123-4567.")
        assert len(dets) == 1
        assert dets[0].entity_type == "email"

    def test_no_match(self):
        s = RegexDetectionStrategy()
        dets = s.detect("Hello world, no PII here.")
        assert dets == []

    def test_empty_string(self):
        s = RegexDetectionStrategy()
        assert s.detect("") == []

    def test_offsets_are_correct(self):
        s = RegexDetectionStrategy(enabled_entity_types=["email"])
        text = "Hello alice@example.com!"
        dets = s.detect(text)
        assert len(dets) == 1
        d = dets[0]
        assert text[d.start:d.end] == "alice@example.com"


# ============================================================================
# FieldHeuristicStrategy
# ============================================================================


class TestFieldHeuristicStrategy:
    def test_name(self):
        s = FieldHeuristicStrategy()
        assert s.name == "field_heuristic"

    def test_sensitive_field_detected(self):
        s = FieldHeuristicStrategy()
        dets = s.detect("Alice Smith", field_path="user.name")
        assert len(dets) == 1
        assert dets[0].value == "Alice Smith"
        assert dets[0].entity_type == "person_name"
        assert dets[0].confidence == 0.9
        assert dets[0].source == "field_heuristic"

    def test_email_field(self):
        s = FieldHeuristicStrategy()
        dets = s.detect("alice@example.com", field_path="contact.email")
        assert len(dets) == 1
        assert dets[0].entity_type == "email"

    def test_non_sensitive_field_ignored(self):
        s = FieldHeuristicStrategy()
        dets = s.detect("some value", field_path="user.favorite_color")
        assert dets == []

    def test_empty_field_path(self):
        s = FieldHeuristicStrategy()
        assert s.detect("Alice", field_path="") == []

    def test_empty_text(self):
        s = FieldHeuristicStrategy()
        assert s.detect("", field_path="user.name") == []

    def test_custom_sensitive_fields(self):
        s = FieldHeuristicStrategy(sensitive_fields={"custom_field"})
        dets = s.detect("value", field_path="data.custom_field")
        assert len(dets) == 1
        # Custom field not in mapping -> "sensitive_field"
        assert dets[0].entity_type == "sensitive_field"

    def test_learned_fields(self):
        s = FieldHeuristicStrategy(learned_fields={"tenant_id"})
        dets = s.detect("12345", field_path="data.tenant_id")
        assert len(dets) == 1

    def test_case_insensitive_leaf(self):
        s = FieldHeuristicStrategy()
        dets = s.detect("John", field_path="user.First_Name")
        assert len(dets) == 1
        assert dets[0].entity_type == "person_name"

    def test_nested_path_uses_leaf(self):
        s = FieldHeuristicStrategy()
        dets = s.detect("555-123-4567", field_path="deeply.nested.path.phone_number")
        assert len(dets) == 1
        assert dets[0].entity_type == "phone"

    def test_span_covers_full_text(self):
        s = FieldHeuristicStrategy()
        text = "Alice Smith"
        dets = s.detect(text, field_path="user.name")
        assert dets[0].start == 0
        assert dets[0].end == len(text)


# ============================================================================
# NERDetectionStrategy
# ============================================================================


class TestNERDetectionStrategy:
    def test_name(self):
        s = NERDetectionStrategy()
        assert s.name == "ner_model"

    def test_disabled_when_transformers_not_installed(self):
        s = NERDetectionStrategy()
        # Force _available to False (simulating missing import)
        s._available = False
        dets = s.detect("John Smith lives in New York.")
        assert dets == []

    def test_skips_long_text(self):
        s = NERDetectionStrategy(max_text_length=10)
        s._available = True
        s._pipeline = MagicMock()
        dets = s.detect("A" * 100)
        assert dets == []
        s._pipeline.assert_not_called()

    def test_skips_empty_text(self):
        s = NERDetectionStrategy()
        s._available = True
        s._pipeline = MagicMock()
        dets = s.detect("")
        assert dets == []

    def test_detect_with_mocked_pipeline(self):
        s = NERDetectionStrategy(confidence_threshold=0.5)
        s._available = True
        s._pipeline = MagicMock(return_value=[
            {
                "entity_group": "PER",
                "score": 0.95,
                "word": "John Smith",
                "start": 0,
                "end": 10,
            },
            {
                "entity_group": "LOC",
                "score": 0.88,
                "word": "New York",
                "start": 20,
                "end": 28,
            },
        ])
        text = "John Smith lives in New York."
        dets = s.detect(text)
        assert len(dets) == 2
        assert dets[0].entity_type == "person_name"
        assert dets[0].value == "John Smith"
        assert dets[0].confidence == 0.95
        assert dets[0].source == "ner_model"
        assert dets[1].entity_type == "location"
        assert dets[1].value == "New York"

    def test_filters_low_confidence(self):
        s = NERDetectionStrategy(confidence_threshold=0.9)
        s._available = True
        s._pipeline = MagicMock(return_value=[
            {"entity_group": "PER", "score": 0.85, "word": "Bob", "start": 0, "end": 3},
            {"entity_group": "PER", "score": 0.95, "word": "Alice", "start": 10, "end": 15},
        ])
        dets = s.detect("Bob and Alice")
        assert len(dets) == 1
        assert dets[0].value == "Alice"

    def test_handles_pipeline_error(self):
        s = NERDetectionStrategy()
        s._available = True
        s._pipeline = MagicMock(side_effect=RuntimeError("model error"))
        dets = s.detect("Some text")
        assert dets == []

    def test_ensure_loaded_import_error(self):
        s = NERDetectionStrategy()
        with patch.dict("sys.modules", {"transformers": None}):
            # Reset so it tries again
            s._available = None
            # The import will fail
            result = s._ensure_loaded()
            assert result is False
            assert s._available is False

    def test_label_mapping(self):
        s = NERDetectionStrategy(confidence_threshold=0.0)
        s._available = True
        s._pipeline = MagicMock(return_value=[
            {"entity_group": "ORG", "score": 0.9, "word": "Acme", "start": 0, "end": 4},
            {"entity_group": "MISC", "score": 0.8, "word": "stuff", "start": 5, "end": 10},
        ])
        dets = s.detect("Acme stuff")
        assert dets[0].entity_type == "organization"
        assert dets[1].entity_type == "misc_entity"

    def test_unknown_label_uses_lowercase(self):
        s = NERDetectionStrategy(confidence_threshold=0.0)
        s._available = True
        s._pipeline = MagicMock(return_value=[
            {"entity_group": "CUSTOM_TYPE", "score": 0.9, "word": "val", "start": 0, "end": 3},
        ])
        dets = s.detect("val")
        assert dets[0].entity_type == "custom_type"


# ============================================================================
# _merge_detections
# ============================================================================


class TestMergeDetections:
    def _det(self, start: int, end: int, conf: float = 1.0, source: str = "regex", entity_type: str = "email") -> PIIDetection:
        return PIIDetection(
            start=start, end=end, value=f"v{start}",
            entity_type=entity_type, confidence=conf, source=source,
        )

    def test_empty(self):
        assert _merge_detections([]) == []

    def test_no_overlap(self):
        dets = [self._det(0, 5), self._det(10, 15)]
        result = _merge_detections(dets)
        assert len(result) == 2

    def test_overlap_keeps_higher_confidence(self):
        d1 = self._det(0, 10, conf=0.7, source="regex")
        d2 = self._det(5, 15, conf=0.9, source="ner_model")
        result = _merge_detections([d1, d2])
        assert len(result) == 1
        assert result[0].confidence == 0.9
        assert result[0].source == "ner_model"

    def test_overlap_keeps_first_when_equal_confidence(self):
        d1 = self._det(0, 10, conf=1.0, source="regex")
        d2 = self._det(5, 15, conf=1.0, source="ner_model")
        result = _merge_detections([d1, d2])
        assert len(result) == 1
        # First in sorted order wins when equal confidence
        assert result[0].source == "regex"

    def test_adjacent_not_merged(self):
        d1 = self._det(0, 5)
        d2 = self._det(5, 10)
        result = _merge_detections([d1, d2])
        assert len(result) == 2

    def test_single_detection(self):
        d = self._det(0, 5)
        result = _merge_detections([d])
        assert len(result) == 1
        assert result[0] == d

    def test_three_way_overlap(self):
        d1 = self._det(0, 10, conf=0.5)
        d2 = self._det(5, 15, conf=0.9)
        d3 = self._det(12, 20, conf=0.8)
        result = _merge_detections([d1, d2, d3])
        # d1 overlaps d2 -> d2 wins (0.9 > 0.5)
        # d2 overlaps d3 -> d2 wins (0.9 > 0.8)
        assert len(result) == 1
        assert result[0].confidence == 0.9


# ============================================================================
# CompositeDetectionPipeline
# ============================================================================


class TestCompositeDetectionPipeline:
    def test_empty_pipeline(self):
        p = CompositeDetectionPipeline()
        assert p.detect_all("hello") == []

    def test_single_strategy(self):
        p = CompositeDetectionPipeline([RegexDetectionStrategy()])
        dets = p.detect_all("Email alice@example.com here.")
        assert len(dets) >= 1
        assert dets[0].entity_type == "email"

    def test_multiple_strategies(self):
        regex = RegexDetectionStrategy()
        field = FieldHeuristicStrategy()
        p = CompositeDetectionPipeline([regex, field])
        # Regex finds the email pattern; field heuristic detects the full value for "email" field
        dets = p.detect_all("alice@example.com", field_path="user.email")
        assert len(dets) >= 1

    def test_add_strategy(self):
        p = CompositeDetectionPipeline()
        p.add_strategy(RegexDetectionStrategy())
        assert len(p.strategies) == 1
        dets = p.detect_all("Call 555-123-4567.")
        assert len(dets) == 1

    def test_error_resilient(self):
        """A failing strategy should not prevent others from running."""
        failing = MagicMock()
        failing.detect = MagicMock(side_effect=RuntimeError("boom"))
        failing.name = "failing"
        regex = RegexDetectionStrategy()
        p = CompositeDetectionPipeline([failing, regex])
        dets = p.detect_all("Contact alice@example.com")
        # Regex should still work despite the failing strategy
        assert len(dets) == 1
        assert dets[0].entity_type == "email"

    def test_merges_overlapping_detections(self):
        """When multiple strategies detect overlapping spans, highest confidence wins."""
        # Create two mock strategies that detect overlapping spans
        s1 = MagicMock()
        s1.name = "s1"
        s1.detect = MagicMock(return_value=[
            PIIDetection(start=0, end=10, value="John Smith", entity_type="person_name", confidence=0.6, source="s1"),
        ])
        s2 = MagicMock()
        s2.name = "s2"
        s2.detect = MagicMock(return_value=[
            PIIDetection(start=0, end=10, value="John Smith", entity_type="person_name", confidence=0.95, source="s2"),
        ])
        p = CompositeDetectionPipeline([s1, s2])
        dets = p.detect_all("John Smith lives here.")
        assert len(dets) == 1
        assert dets[0].confidence == 0.95
        assert dets[0].source == "s2"

    def test_field_path_passed_to_strategies(self):
        mock = MagicMock()
        mock.name = "mock"
        mock.detect = MagicMock(return_value=[])
        p = CompositeDetectionPipeline([mock])
        p.detect_all("text", field_path="user.email")
        mock.detect.assert_called_once_with("text", "user.email")

    def test_strategies_property(self):
        r = RegexDetectionStrategy()
        f = FieldHeuristicStrategy()
        p = CompositeDetectionPipeline([r, f])
        assert len(p.strategies) == 2
