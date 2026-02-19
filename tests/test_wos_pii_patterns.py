"""Tests for apprentice.wos_pii_patterns — WOS-specific PII pattern extensions."""

import re
import pytest

from apprentice.wos_pii_patterns import (
    WOS_CUSTOM_PATTERNS,
    BUILTIN_ENTITY_TYPES,
    create_wos_pii_tokenizer,
    register_wos_pii_tokenizer,
    validate_wos_patterns,
)
from apprentice.pii_tokenizer import PIITokenizer, PIITokenizerConfig
from apprentice.plugin_registry import PluginRegistrySet, DuplicatePluginError
from apprentice.middleware import MiddlewareContext, MiddlewareResponse


WOS_PATTERN_NAMES = list(WOS_CUSTOM_PATTERNS.keys())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tokenizer():
    return create_wos_pii_tokenizer()


@pytest.fixture
def registry():
    return PluginRegistrySet.with_defaults()


# ---------------------------------------------------------------------------
# WOS_CUSTOM_PATTERNS constant
# ---------------------------------------------------------------------------

class TestWOSCustomPatterns:
    def test_has_10_entries(self):
        assert len(WOS_CUSTOM_PATTERNS) == 10

    def test_all_values_are_strings(self):
        for name, pattern in WOS_CUSTOM_PATTERNS.items():
            assert isinstance(pattern, str), f"Pattern '{name}' value must be a string"

    def test_all_compile(self):
        for name, pattern in WOS_CUSTOM_PATTERNS.items():
            try:
                re.compile(pattern)
            except re.error as e:
                pytest.fail(f"Pattern '{name}' failed to compile: {e}")

    def test_no_collision_with_builtins(self):
        overlap = set(BUILTIN_ENTITY_TYPES) & set(WOS_CUSTOM_PATTERNS.keys())
        assert len(overlap) == 0, f"Collision: {overlap}"

    def test_expected_pattern_names(self):
        expected = {
            "wander_booking_id", "stripe_payment_intent", "stripe_charge",
            "stripe_refund", "stripe_customer", "guesty_reservation_id",
            "hostaway_reservation_id", "ownerrez_reservation_id",
            "streamline_reservation_id", "us_street_address",
        }
        assert set(WOS_CUSTOM_PATTERNS.keys()) == expected


# ---------------------------------------------------------------------------
# Pattern matching — positive samples
# ---------------------------------------------------------------------------

POSITIVE_SAMPLES = {
    "wander_booking_id": ["W-AB12CD34", "W-ZZZZ9999"],
    "stripe_payment_intent": ["pi_3MtwBwLkdIwHu7ix28a3tqPa", "pi_abcdefghijklmnopqrstuvwx"],
    "stripe_charge": ["ch_3MtwBwLkdIwHu7ix28a3tqPa", "ch_abcdefghijklmnopqrstuvwx"],
    "stripe_refund": ["re_3MtwBwLkdIwHu7ix28a3tqPa", "re_abcdefghijklmnopqrstuvwx"],
    "stripe_customer": ["cus_9s6XKzkNRiz8i3", "cus_AbCdEfGhIjKlMnOpQr"],
    "guesty_reservation_id": ["507f1f77bcf86cd799439011", "60b8d295f0d8a82e3c000001"],
    "hostaway_reservation_id": ["123456", "1234567890"],
    "ownerrez_reservation_id": ["OR-ABC123", "OR-XY9876ZZ"],
    "streamline_reservation_id": ["SL12345678", "SL123456789"],
    "us_street_address": ["123 Main St", "4567 Oak Avenue"],
}


@pytest.mark.parametrize("pattern_name", WOS_PATTERN_NAMES)
def test_pattern_positive_match(pattern_name):
    compiled = re.compile(WOS_CUSTOM_PATTERNS[pattern_name])
    for sample in POSITIVE_SAMPLES.get(pattern_name, []):
        assert compiled.search(sample) is not None, \
            f"Pattern '{pattern_name}' should match '{sample}'"


# ---------------------------------------------------------------------------
# Pattern matching — negative samples
# ---------------------------------------------------------------------------

NEGATIVE_SAMPLES = {
    "wander_booking_id": ["W-abc12cd3", "W-SHORT", "WND-123456"],
    "stripe_payment_intent": ["pi_short", "PI_3MtwBwLkdIwHu7ix28a3tqPa"],
    "stripe_charge": ["ch_short", "CH_3MtwBwLkdIwHu7ix28a3tqPa"],
    "stripe_refund": ["re_short", "RE_3MtwBwLkdIwHu7ix28a3tqPa"],
    "stripe_customer": ["cus_short", "CUS_9s6XKzkNRiz8i3"],
    "guesty_reservation_id": ["507f1f77bcf86cd79943901", "NOTAHEXSTRING12345678901"],
    "hostaway_reservation_id": ["12345", "12345678901"],
    "ownerrez_reservation_id": ["OR-AB", "or-ABC123"],
    "streamline_reservation_id": ["SL1234567", "sl12345678"],
    "us_street_address": ["no address here", "just numbers 12345"],
}


@pytest.mark.parametrize("pattern_name", WOS_PATTERN_NAMES)
def test_pattern_negative_match(pattern_name):
    compiled = re.compile(WOS_CUSTOM_PATTERNS[pattern_name])
    for sample in NEGATIVE_SAMPLES.get(pattern_name, []):
        assert compiled.search(sample) is None, \
            f"Pattern '{pattern_name}' should NOT match '{sample}'"


# ---------------------------------------------------------------------------
# create_wos_pii_tokenizer — happy path
# ---------------------------------------------------------------------------

class TestCreateFactory:
    def test_returns_pii_tokenizer(self, tokenizer):
        assert isinstance(tokenizer, PIITokenizer)

    def test_default_has_all_wos_patterns(self, tokenizer):
        config = tokenizer._config
        for name in WOS_PATTERN_NAMES:
            assert name in config.custom_patterns

    def test_default_has_all_entity_types(self, tokenizer):
        types = tokenizer._config.enabled_entity_types
        for bt in BUILTIN_ENTITY_TYPES:
            assert bt in types
        for wn in WOS_PATTERN_NAMES:
            assert wn in types

    def test_default_has_14_entity_types(self, tokenizer):
        assert len(tokenizer._config.enabled_entity_types) == 14

    def test_custom_entity_types(self):
        t = create_wos_pii_tokenizer(enabled_entity_types=["email", "phone"])
        assert list(t._config.enabled_entity_types) == ["email", "phone"]

    def test_additional_patterns_merged(self):
        t = create_wos_pii_tokenizer(additional_patterns={"my_custom": r"CUSTOM-\d{4}"})
        assert "my_custom" in t._config.custom_patterns
        for name in WOS_PATTERN_NAMES:
            assert name in t._config.custom_patterns

    def test_additional_patterns_override(self):
        override = r"OVERRIDE-\d+"
        t = create_wos_pii_tokenizer(additional_patterns={"wander_booking_id": override})
        assert t._config.custom_patterns["wander_booking_id"] == override

    def test_custom_token_format(self):
        fmt = "<<{type}:{hash}>>"
        t = create_wos_pii_tokenizer(token_format=fmt)
        assert t._config.token_format == fmt

    def test_empty_additional_patterns(self):
        t = create_wos_pii_tokenizer(additional_patterns={})
        assert len(t._config.custom_patterns) >= 10


# ---------------------------------------------------------------------------
# create_wos_pii_tokenizer — error cases
# ---------------------------------------------------------------------------

class TestCreateFactoryErrors:
    def test_invalid_format_missing_type(self):
        with pytest.raises(ValueError):
            create_wos_pii_tokenizer(token_format="[{hash}]")

    def test_invalid_format_missing_hash(self):
        with pytest.raises(ValueError):
            create_wos_pii_tokenizer(token_format="[{type}]")

    def test_invalid_format_missing_both(self):
        with pytest.raises(ValueError):
            create_wos_pii_tokenizer(token_format="REDACTED")

    def test_invalid_regex_in_additional(self):
        with pytest.raises(re.error):
            create_wos_pii_tokenizer(additional_patterns={"bad": r"[invalid"})

    def test_empty_string_entity_type(self):
        with pytest.raises(ValueError):
            create_wos_pii_tokenizer(enabled_entity_types=["email", ""])

    def test_non_string_entity_type(self):
        with pytest.raises(ValueError):
            create_wos_pii_tokenizer(enabled_entity_types=["email", 123])


# ---------------------------------------------------------------------------
# Middleware protocol roundtrip
# ---------------------------------------------------------------------------

class TestMiddlewareRoundtrip:
    def test_pre_process_post_process(self, tokenizer):
        ctx = MiddlewareContext(
            request_id="test-1",
            task_name="test",
            input_data={"text": "Booking W-AB12CD34 for guest"},
        )
        processed = tokenizer.pre_process(ctx)
        assert "W-AB12CD34" not in str(processed.input_data)
        assert "<PII:" in str(processed.input_data)

        resp = MiddlewareResponse(output_data=processed.input_data)
        restored = tokenizer.post_process(processed, resp)
        assert "W-AB12CD34" in str(restored.output_data)

    def test_no_pii_passthrough(self, tokenizer):
        ctx = MiddlewareContext(
            request_id="test-2",
            task_name="test",
            input_data={"text": "Hello, normal message"},
        )
        processed = tokenizer.pre_process(ctx)
        assert processed.input_data == ctx.input_data

    def test_multiple_pii_roundtrip(self, tokenizer):
        text = "Booking W-AB12CD34 paid via pi_3MtwBwLkdIwHu7ix28a3tqPa for cus_AbCdEfGhIjKlMnOpQr"
        ctx = MiddlewareContext(
            request_id="test-3",
            task_name="test",
            input_data={"text": text},
        )
        processed = tokenizer.pre_process(ctx)
        assert "W-AB12CD34" not in str(processed.input_data)

        resp = MiddlewareResponse(output_data=processed.input_data)
        restored = tokenizer.post_process(processed, resp)
        assert restored.output_data["text"] == text

    def test_nested_dict_pii(self, tokenizer):
        ctx = MiddlewareContext(
            request_id="test-4",
            task_name="test",
            input_data={
                "conversation": {
                    "messages": [
                        {"text": "My booking is W-AB12CD34"},
                        {"text": "Charge ch_3MtwBwLkdIwHu7ix28a3tqPa processed"},
                    ]
                }
            },
        )
        processed = tokenizer.pre_process(ctx)
        flat = str(processed.input_data)
        assert "W-AB12CD34" not in flat
        assert "ch_3MtwBwLkdIwHu7ix28a3tqPa" not in flat

    def test_stripe_ids_tokenized(self, tokenizer):
        ctx = MiddlewareContext(
            request_id="test-5",
            task_name="test",
            input_data={"payment": "pi_3MtwBwLkdIwHu7ix28a3tqPa"},
        )
        processed = tokenizer.pre_process(ctx)
        assert "pi_3MtwBwLkdIwHu7ix28a3tqPa" not in str(processed.input_data)


# ---------------------------------------------------------------------------
# register_wos_pii_tokenizer
# ---------------------------------------------------------------------------

class TestPluginRegistration:
    def test_register_on_fresh_registry(self, registry):
        register_wos_pii_tokenizer(registry)
        mw = registry.get_registry("middleware")
        assert "wos_pii_tokenizer" in mw

    def test_create_from_registry(self, registry):
        register_wos_pii_tokenizer(registry)
        mw = registry.get_registry("middleware")
        tokenizer = mw.create("wos_pii_tokenizer")
        assert isinstance(tokenizer, PIITokenizer)

    def test_duplicate_raises(self, registry):
        register_wos_pii_tokenizer(registry)
        with pytest.raises(DuplicatePluginError):
            register_wos_pii_tokenizer(registry)


# ---------------------------------------------------------------------------
# validate_wos_patterns
# ---------------------------------------------------------------------------

class TestValidatePatterns:
    def test_returns_dict_with_10_keys(self):
        result = validate_wos_patterns()
        assert isinstance(result, dict)
        assert len(result) == 10

    def test_all_keys_present(self):
        result = validate_wos_patterns()
        for name in WOS_PATTERN_NAMES:
            assert name in result

    def test_all_values_true(self):
        result = validate_wos_patterns()
        for name, ok in result.items():
            assert ok is True, f"Pattern '{name}' should compile successfully"

    def test_never_raises(self):
        validate_wos_patterns()
