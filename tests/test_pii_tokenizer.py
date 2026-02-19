"""Tests for the PII tokenizer middleware."""

import os
import tempfile

import pytest
import yaml

from apprentice.middleware import MiddlewareContext, MiddlewareResponse
from apprentice.pii_tokenizer import (
    FieldPIIStats,
    LearnedPatternStore,
    PIITokenizer,
    PIITokenizerConfig,
    TokenRegistry,
)


# ============================================================================
# PIITokenizerConfig
# ============================================================================


class TestPIITokenizerConfig:
    def test_defaults(self):
        config = PIITokenizerConfig()
        assert config.enabled_entity_types == [
            "email", "phone", "ssn", "credit_card", "ip_address", "api_key",
        ]
        assert config.custom_patterns == {}
        assert config.token_format == "<PII:{type}:{hash}>"
        assert "password" in config.sensitive_fields

    def test_custom_config(self):
        config = PIITokenizerConfig(
            enabled_entity_types=["email"],
            custom_patterns={"ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"},
            token_format="[REDACTED:{type}:{hash}]",
        )
        assert config.enabled_entity_types == ["email"]
        assert "ip_address" in config.custom_patterns
        assert config.token_format == "[REDACTED:{type}:{hash}]"

    def test_frozen(self):
        config = PIITokenizerConfig()
        with pytest.raises(Exception):
            config.enabled_entity_types = ["email"]


# ============================================================================
# TokenRegistry
# ============================================================================


class TestTokenRegistry:
    def test_tokenize_detokenize_roundtrip(self):
        registry = TokenRegistry()
        token = registry.tokenize("test@example.com", "email")
        assert token != "test@example.com"
        assert registry.detokenize(token) == "test@example.com"

    def test_deterministic_same_input(self):
        registry = TokenRegistry()
        token1 = registry.tokenize("test@example.com", "email")
        token2 = registry.tokenize("test@example.com", "email")
        assert token1 == token2

    def test_different_values_different_tokens(self):
        registry = TokenRegistry()
        token1 = registry.tokenize("alice@example.com", "email")
        token2 = registry.tokenize("bob@example.com", "email")
        assert token1 != token2

    def test_token_format(self):
        registry = TokenRegistry(token_format="<PII:{type}:{hash}>")
        token = registry.tokenize("test@example.com", "email")
        assert token.startswith("<PII:email:")
        assert token.endswith(">")

    def test_custom_token_format(self):
        registry = TokenRegistry(token_format="[REDACTED:{type}:{hash}]")
        token = registry.tokenize("555-12-3456", "ssn")
        assert token.startswith("[REDACTED:ssn:")
        assert token.endswith("]")

    def test_clear(self):
        registry = TokenRegistry()
        token = registry.tokenize("test@example.com", "email")
        registry.clear()
        # After clear, detokenize should return the token itself (unrecognized)
        assert registry.detokenize(token) == token

    def test_detokenize_unknown_returns_unchanged(self):
        registry = TokenRegistry()
        assert registry.detokenize("not-a-token") == "not-a-token"


# ============================================================================
# PIITokenizer — pre_process
# ============================================================================


class TestPIITokenizerPreProcess:
    def _make_context(self, input_data: dict) -> MiddlewareContext:
        return MiddlewareContext(
            request_id="req-001",
            task_name="test_task",
            input_data=input_data,
        )

    def test_pre_process_email(self):
        tokenizer = PIITokenizer()
        ctx = self._make_context({"text": "Contact alice@example.com for info."})
        result = tokenizer.pre_process(ctx)
        assert "alice@example.com" not in result.input_data["text"]
        assert "<PII:email:" in result.input_data["text"]
        assert "pii_token_registry" in result.middleware_state

    def test_pre_process_phone(self):
        tokenizer = PIITokenizer()
        ctx = self._make_context({"text": "Call 555-123-4567 today."})
        result = tokenizer.pre_process(ctx)
        assert "555-123-4567" not in result.input_data["text"]
        assert "<PII:phone:" in result.input_data["text"]

    def test_pre_process_phone_no_dashes(self):
        tokenizer = PIITokenizer()
        ctx = self._make_context({"text": "Call 5551234567 today."})
        result = tokenizer.pre_process(ctx)
        assert "5551234567" not in result.input_data["text"]
        assert "<PII:phone:" in result.input_data["text"]

    def test_pre_process_phone_dots(self):
        tokenizer = PIITokenizer()
        ctx = self._make_context({"text": "Call 555.123.4567 today."})
        result = tokenizer.pre_process(ctx)
        assert "555.123.4567" not in result.input_data["text"]
        assert "<PII:phone:" in result.input_data["text"]

    def test_pre_process_ssn(self):
        tokenizer = PIITokenizer()
        ctx = self._make_context({"text": "SSN is 123-45-6789."})
        result = tokenizer.pre_process(ctx)
        assert "123-45-6789" not in result.input_data["text"]
        assert "<PII:ssn:" in result.input_data["text"]

    def test_pre_process_credit_card(self):
        tokenizer = PIITokenizer()
        ctx = self._make_context({"text": "Card: 4111-1111-1111-1111."})
        result = tokenizer.pre_process(ctx)
        assert "4111-1111-1111-1111" not in result.input_data["text"]
        assert "<PII:credit_card:" in result.input_data["text"]

    def test_pre_process_credit_card_spaces(self):
        tokenizer = PIITokenizer()
        ctx = self._make_context({"text": "Card: 4111 1111 1111 1111."})
        result = tokenizer.pre_process(ctx)
        assert "4111 1111 1111 1111" not in result.input_data["text"]
        assert "<PII:credit_card:" in result.input_data["text"]

    def test_pre_process_credit_card_no_separators(self):
        tokenizer = PIITokenizer()
        ctx = self._make_context({"text": "Card: 4111111111111111."})
        result = tokenizer.pre_process(ctx)
        assert "4111111111111111" not in result.input_data["text"]
        assert "<PII:credit_card:" in result.input_data["text"]

    def test_pre_process_custom_patterns(self):
        config = PIITokenizerConfig(
            enabled_entity_types=[],
            custom_patterns={"ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"},
        )
        tokenizer = PIITokenizer(config)
        ctx = self._make_context({"text": "Server at 192.168.1.1 is down."})
        result = tokenizer.pre_process(ctx)
        assert "192.168.1.1" not in result.input_data["text"]
        assert "<PII:ip_address:" in result.input_data["text"]


# ============================================================================
# PIITokenizer — post_process
# ============================================================================


class TestPIITokenizerPostProcess:
    def test_post_process_restores_original_values(self):
        tokenizer = PIITokenizer()
        ctx = MiddlewareContext(
            request_id="req-001",
            task_name="test_task",
            input_data={"text": "Contact alice@example.com for info."},
        )
        pre_result = tokenizer.pre_process(ctx)

        # Simulate model echoing back the tokenized value
        response = MiddlewareResponse(
            output_data={"reply": pre_result.input_data["text"]},
            middleware_state=pre_result.middleware_state,
        )
        post_result = tokenizer.post_process(pre_result, response)
        assert "alice@example.com" in post_result.output_data["reply"]

    def test_post_process_missing_registry_passes_through(self):
        tokenizer = PIITokenizer()
        ctx = MiddlewareContext(
            request_id="req-001",
            task_name="test_task",
            input_data={},
            middleware_state={},
        )
        response = MiddlewareResponse(
            output_data={"reply": "No PII here."},
        )
        post_result = tokenizer.post_process(ctx, response)
        assert post_result.output_data["reply"] == "No PII here."


# ============================================================================
# PIITokenizer — round-trip
# ============================================================================


class TestPIITokenizerRoundTrip:
    def test_round_trip_preserves_data(self):
        tokenizer = PIITokenizer()
        original_text = "Email alice@example.com or call 555-123-4567."
        ctx = MiddlewareContext(
            request_id="req-001",
            task_name="test_task",
            input_data={"text": original_text},
        )

        # pre_process tokenizes
        pre_result = tokenizer.pre_process(ctx)
        assert "alice@example.com" not in pre_result.input_data["text"]
        assert "555-123-4567" not in pre_result.input_data["text"]

        # Simulate model echoing back the tokenized text
        response = MiddlewareResponse(
            output_data={"reply": pre_result.input_data["text"]},
            middleware_state=pre_result.middleware_state,
        )

        # post_process restores
        post_result = tokenizer.post_process(pre_result, response)
        assert post_result.output_data["reply"] == original_text

    def test_round_trip_with_multiple_pii(self):
        tokenizer = PIITokenizer()
        original_text = (
            "User alice@example.com has SSN 123-45-6789 "
            "and card 4111-1111-1111-1111."
        )
        ctx = MiddlewareContext(
            request_id="req-002",
            task_name="test_task",
            input_data={"text": original_text},
        )

        pre_result = tokenizer.pre_process(ctx)
        assert "alice@example.com" not in pre_result.input_data["text"]
        assert "123-45-6789" not in pre_result.input_data["text"]
        assert "4111-1111-1111-1111" not in pre_result.input_data["text"]

        response = MiddlewareResponse(
            output_data={"reply": pre_result.input_data["text"]},
            middleware_state=pre_result.middleware_state,
        )

        post_result = tokenizer.post_process(pre_result, response)
        assert post_result.output_data["reply"] == original_text


# ============================================================================
# PIITokenizer — nested dict input_data
# ============================================================================


class TestPIITokenizerNestedData:
    def test_nested_dict(self):
        tokenizer = PIITokenizer()
        ctx = MiddlewareContext(
            request_id="req-001",
            task_name="test_task",
            input_data={
                "user": {
                    "name": "Alice",
                    "email": "alice@example.com",
                    "contacts": {
                        "phone": "555-123-4567",
                    },
                },
                "notes": "SSN: 123-45-6789",
            },
        )
        pre_result = tokenizer.pre_process(ctx)
        assert "alice@example.com" not in pre_result.input_data["user"]["email"]
        assert "555-123-4567" not in pre_result.input_data["user"]["contacts"]["phone"]
        assert "123-45-6789" not in pre_result.input_data["notes"]

        # Round-trip restores
        response = MiddlewareResponse(
            output_data=pre_result.input_data,
            middleware_state=pre_result.middleware_state,
        )
        post_result = tokenizer.post_process(pre_result, response)
        assert post_result.output_data["user"]["email"] == "alice@example.com"
        assert post_result.output_data["user"]["contacts"]["phone"] == "555-123-4567"
        assert "123-45-6789" in post_result.output_data["notes"]

    def test_nested_list(self):
        tokenizer = PIITokenizer()
        ctx = MiddlewareContext(
            request_id="req-001",
            task_name="test_task",
            input_data={
                "emails": ["alice@example.com", "bob@example.com"],
            },
        )
        pre_result = tokenizer.pre_process(ctx)
        for item in pre_result.input_data["emails"]:
            assert "@example.com" not in item
            assert "<PII:email:" in item

        # Round-trip restores
        response = MiddlewareResponse(
            output_data=pre_result.input_data,
            middleware_state=pre_result.middleware_state,
        )
        post_result = tokenizer.post_process(pre_result, response)
        assert post_result.output_data["emails"] == [
            "alice@example.com",
            "bob@example.com",
        ]


# ============================================================================
# PIITokenizer — no PII passthrough
# ============================================================================


class TestPIITokenizerNoPII:
    def test_no_pii_passthrough(self):
        tokenizer = PIITokenizer()
        ctx = MiddlewareContext(
            request_id="req-001",
            task_name="test_task",
            input_data={"text": "Hello world, no sensitive data here."},
        )
        pre_result = tokenizer.pre_process(ctx)
        assert pre_result.input_data["text"] == "Hello world, no sensitive data here."
        assert "pii_token_registry" in pre_result.middleware_state

    def test_no_pii_numeric_values_passthrough(self):
        tokenizer = PIITokenizer()
        ctx = MiddlewareContext(
            request_id="req-001",
            task_name="test_task",
            input_data={"count": 42, "flag": True, "value": None},
        )
        pre_result = tokenizer.pre_process(ctx)
        assert pre_result.input_data["count"] == 42
        assert pre_result.input_data["flag"] is True
        assert pre_result.input_data["value"] is None


# ============================================================================
# PIITokenizer — disabled entity types
# ============================================================================


class TestPIITokenizerDisabledTypes:
    def test_disabled_email_not_tokenized(self):
        config = PIITokenizerConfig(enabled_entity_types=["phone"])
        tokenizer = PIITokenizer(config)
        ctx = MiddlewareContext(
            request_id="req-001",
            task_name="test_task",
            input_data={"text": "Email alice@example.com or call 555-123-4567."},
        )
        pre_result = tokenizer.pre_process(ctx)
        # Email should NOT be tokenized (not enabled)
        assert "alice@example.com" in pre_result.input_data["text"]
        # Phone SHOULD be tokenized
        assert "555-123-4567" not in pre_result.input_data["text"]
        assert "<PII:phone:" in pre_result.input_data["text"]

    def test_all_types_disabled(self):
        config = PIITokenizerConfig(enabled_entity_types=[])
        tokenizer = PIITokenizer(config)
        ctx = MiddlewareContext(
            request_id="req-001",
            task_name="test_task",
            input_data={
                "text": "Email alice@example.com, SSN 123-45-6789, phone 555-123-4567."
            },
        )
        pre_result = tokenizer.pre_process(ctx)
        # Nothing should be tokenized
        assert "alice@example.com" in pre_result.input_data["text"]
        assert "123-45-6789" in pre_result.input_data["text"]
        assert "555-123-4567" in pre_result.input_data["text"]

    def test_only_ssn_enabled(self):
        config = PIITokenizerConfig(enabled_entity_types=["ssn"])
        tokenizer = PIITokenizer(config)
        ctx = MiddlewareContext(
            request_id="req-001",
            task_name="test_task",
            input_data={"text": "SSN 123-45-6789, email alice@example.com."},
        )
        pre_result = tokenizer.pre_process(ctx)
        assert "123-45-6789" not in pre_result.input_data["text"]
        assert "<PII:ssn:" in pre_result.input_data["text"]
        assert "alice@example.com" in pre_result.input_data["text"]


# ============================================================================
# New built-in patterns: ip_address and api_key
# ============================================================================


class TestPIITokenizerNewPatterns:
    def _make_context(self, input_data: dict) -> MiddlewareContext:
        return MiddlewareContext(
            request_id="req-001",
            task_name="test_task",
            input_data=input_data,
        )

    def test_ip_address_tokenized(self):
        tokenizer = PIITokenizer()
        ctx = self._make_context({"text": "Server at 192.168.1.100 is down."})
        result = tokenizer.pre_process(ctx)
        assert "192.168.1.100" not in result.input_data["text"]
        assert "<PII:ip_address:" in result.input_data["text"]

    def test_ip_address_roundtrip(self):
        tokenizer = PIITokenizer()
        ctx = self._make_context({"text": "Connect to 10.0.0.1 now."})
        pre = tokenizer.pre_process(ctx)
        resp = MiddlewareResponse(
            output_data={"reply": pre.input_data["text"]},
            middleware_state=pre.middleware_state,
        )
        post = tokenizer.post_process(pre, resp)
        assert post.output_data["reply"] == "Connect to 10.0.0.1 now."

    def test_api_key_tokenized(self):
        tokenizer = PIITokenizer()
        ctx = self._make_context({"text": "Use key sk-abcdefghijklmnopqrstuvwx to authenticate."})
        result = tokenizer.pre_process(ctx)
        assert "sk-abcdefghijklmnopqrstuvwx" not in result.input_data["text"]
        assert "<PII:api_key:" in result.input_data["text"]

    def test_api_key_various_prefixes(self):
        tokenizer = PIITokenizer()
        keys = [
            "pk-ABCDEFGHIJKLMNOPQRSTUVWX",
            "token-abcdefghijklmnopqrstuv",
            "api_key1234567890abcdefghij",
            "key_abcdefghijklmnopqrstuvwx",
        ]
        for key in keys:
            ctx = self._make_context({"text": f"Key: {key}"})
            result = tokenizer.pre_process(ctx)
            assert key not in result.input_data["text"], f"Expected {key} to be tokenized"


# ============================================================================
# Sensitive fields — field-path based tokenization
# ============================================================================


class TestPIITokenizerSensitiveFields:
    def _make_context(self, input_data: dict) -> MiddlewareContext:
        return MiddlewareContext(
            request_id="req-001",
            task_name="test_task",
            input_data=input_data,
        )

    def test_sensitive_field_tokenized_regardless_of_content(self):
        config = PIITokenizerConfig(
            enabled_entity_types=[],
            sensitive_fields=["password"],
        )
        tokenizer = PIITokenizer(config)
        ctx = self._make_context({"password": "my_secret_pass123"})
        result = tokenizer.pre_process(ctx)
        assert "my_secret_pass123" not in result.input_data["password"]
        assert "<PII:sensitive_field:" in result.input_data["password"]

    def test_sensitive_field_roundtrip(self):
        config = PIITokenizerConfig(
            enabled_entity_types=[],
            sensitive_fields=["password"],
        )
        tokenizer = PIITokenizer(config)
        ctx = self._make_context({"password": "super_secret", "name": "Alice"})
        pre = tokenizer.pre_process(ctx)
        assert "super_secret" not in pre.input_data["password"]
        assert pre.input_data["name"] == "Alice"

        resp = MiddlewareResponse(
            output_data=pre.input_data,
            middleware_state=pre.middleware_state,
        )
        post = tokenizer.post_process(pre, resp)
        assert post.output_data["password"] == "super_secret"
        assert post.output_data["name"] == "Alice"

    def test_nested_sensitive_field(self):
        config = PIITokenizerConfig(
            enabled_entity_types=[],
            sensitive_fields=["secret"],
        )
        tokenizer = PIITokenizer(config)
        ctx = self._make_context({
            "credentials": {"secret": "abc123xyz"},
            "name": "test",
        })
        result = tokenizer.pre_process(ctx)
        assert "abc123xyz" not in result.input_data["credentials"]["secret"]
        assert result.input_data["name"] == "test"

    def test_non_sensitive_field_not_tokenized(self):
        config = PIITokenizerConfig(
            enabled_entity_types=[],
            sensitive_fields=["password"],
        )
        tokenizer = PIITokenizer(config)
        ctx = self._make_context({"username": "alice", "password": "pass"})
        result = tokenizer.pre_process(ctx)
        assert result.input_data["username"] == "alice"
        assert "pass" not in result.input_data["password"] or "<PII:" in result.input_data["password"]


# ============================================================================
# LearnedPatternStore
# ============================================================================


class TestLearnedPatternStore:
    def test_record_and_get_known_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "learned.yaml")
            store = LearnedPatternStore(path=path)
            store.record_match("user.email", "email")
            store.record_match("user.email", "email")
            store.record_match("user.phone", "phone")

            known = store.get_known_fields()
            assert ("user.email", "email") in known
            assert ("user.phone", "phone") in known

    def test_flush_writes_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "learned.yaml")
            store = LearnedPatternStore(path=path)
            store.record_match("user.email", "email")
            store.flush()

            data = yaml.safe_load(open(path))
            assert "learned_fields" in data
            assert len(data["learned_fields"]) == 1
            assert data["learned_fields"][0]["field_path"] == "user.email"
            assert data["learned_fields"][0]["entity_type"] == "email"
            assert data["learned_fields"][0]["match_count"] == 1

    def test_load_persisted_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "learned.yaml")
            # Write initial data
            store1 = LearnedPatternStore(path=path)
            for _ in range(25):
                store1.record_match("user.email", "email")
            store1.flush()

            # Load in new instance
            store2 = LearnedPatternStore(path=path)
            known = store2.get_known_fields()
            assert ("user.email", "email") in known
            high = store2.get_high_confidence_fields()
            assert ("user.email", "email") in high

    def test_confidence_levels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "learned.yaml")
            store = LearnedPatternStore(path=path)

            # Low confidence (< 5 matches)
            store.record_match("a.b", "email")
            assert store.get_high_confidence_fields() == []

            # Medium confidence (5-19 matches)
            for _ in range(4):
                store.record_match("a.b", "email")
            assert ("a.b", "email") not in store.get_high_confidence_fields()

            # High confidence (>= 20 matches)
            for _ in range(15):
                store.record_match("a.b", "email")
            assert ("a.b", "email") in store.get_high_confidence_fields()

    def test_empty_file_no_crash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "learned.yaml")
            # Write empty file
            open(path, "w").close()
            store = LearnedPatternStore(path=path)
            assert store.get_known_fields() == []

    def test_nonexistent_file_no_crash(self):
        store = LearnedPatternStore(path="/tmp/nonexistent_pii_test.yaml")
        assert store.get_known_fields() == []


# ============================================================================
# Field-path tracking with learned store integration
# ============================================================================


class TestPIITokenizerFieldPathTracking:
    def test_field_path_recorded_on_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "learned.yaml")
            store = LearnedPatternStore(path=path)
            tokenizer = PIITokenizer(
                config=PIITokenizerConfig(sensitive_fields=[]),
                learned_store=store,
            )
            ctx = MiddlewareContext(
                request_id="req-001",
                task_name="test_task",
                input_data={"user": {"email": "alice@example.com"}},
            )
            tokenizer.pre_process(ctx)
            known = store.get_known_fields()
            assert ("user.email", "email") in known

    def test_sensitive_field_recorded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "learned.yaml")
            store = LearnedPatternStore(path=path)
            tokenizer = PIITokenizer(
                config=PIITokenizerConfig(
                    enabled_entity_types=[],
                    sensitive_fields=["password"],
                ),
                learned_store=store,
            )
            ctx = MiddlewareContext(
                request_id="req-001",
                task_name="test_task",
                input_data={"password": "secret123"},
            )
            tokenizer.pre_process(ctx)
            known = store.get_known_fields()
            assert ("password", "sensitive_field") in known
