"""Tests for plugin_registry module."""

import threading
import pytest
from types import SimpleNamespace

from apprentice.plugin_registry import (
    PluginRegistry,
    PluginRegistrySet,
    PluginError,
    DuplicatePluginError,
    UnknownPluginError,
    PluginConfigError,
    _validate_plugin_name,
    _make_default_factory,
)


# ============================================================================
# _validate_plugin_name
# ============================================================================


class TestValidatePluginName:
    def test_valid_simple(self):
        _validate_plugin_name("exact_match")

    def test_valid_with_digits(self):
        _validate_plugin_name("backend2")

    def test_valid_single_char(self):
        _validate_plugin_name("a")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _validate_plugin_name("")

    def test_starts_with_digit_raises(self):
        with pytest.raises(ValueError, match="must match"):
            _validate_plugin_name("1backend")

    def test_uppercase_raises(self):
        with pytest.raises(ValueError, match="must match"):
            _validate_plugin_name("ExactMatch")

    def test_hyphen_raises(self):
        with pytest.raises(ValueError, match="must match"):
            _validate_plugin_name("exact-match")

    def test_dot_raises(self):
        with pytest.raises(ValueError, match="must match"):
            _validate_plugin_name("exact.match")

    def test_starts_with_underscore_raises(self):
        with pytest.raises(ValueError, match="must match"):
            _validate_plugin_name("_private")


# ============================================================================
# PluginRegistry
# ============================================================================


class TestPluginRegistry:
    def test_create_with_domain(self):
        r = PluginRegistry("evaluators")
        assert r.domain == "evaluators"

    def test_empty_domain_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            PluginRegistry("")

    def test_whitespace_domain_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            PluginRegistry("   ")

    def test_register_and_create(self):
        r = PluginRegistry("test")
        r.register("my_plugin", lambda **kw: SimpleNamespace(**kw))
        result = r.create("my_plugin", key="value")
        assert result.key == "value"

    def test_register_duplicate_raises(self):
        r = PluginRegistry("test")
        r.register("plugin_a", lambda **kw: None)
        with pytest.raises(DuplicatePluginError) as exc_info:
            r.register("plugin_a", lambda **kw: None)
        assert exc_info.value.name == "plugin_a"
        assert exc_info.value.domain == "test"

    def test_create_unknown_raises(self):
        r = PluginRegistry("test")
        r.register("known", lambda **kw: None)
        with pytest.raises(UnknownPluginError) as exc_info:
            r.create("unknown")
        assert exc_info.value.name == "unknown"
        assert exc_info.value.domain == "test"
        assert "known" in exc_info.value.available

    def test_register_non_callable_raises(self):
        r = PluginRegistry("test")
        with pytest.raises(TypeError, match="callable"):
            r.register("bad", "not_callable")

    def test_register_invalid_name_raises(self):
        r = PluginRegistry("test")
        with pytest.raises(ValueError):
            r.register("Bad-Name", lambda **kw: None)

    def test_validate_name_true(self):
        r = PluginRegistry("test")
        r.register("exists", lambda **kw: None)
        assert r.validate_name("exists") is True

    def test_validate_name_false(self):
        r = PluginRegistry("test")
        assert r.validate_name("missing") is False

    def test_contains(self):
        r = PluginRegistry("test")
        r.register("exists", lambda **kw: None)
        assert "exists" in r
        assert "missing" not in r

    def test_len(self):
        r = PluginRegistry("test")
        assert len(r) == 0
        r.register("a", lambda **kw: None)
        assert len(r) == 1
        r.register("b", lambda **kw: None)
        assert len(r) == 2

    def test_list_plugins_sorted(self):
        r = PluginRegistry("test")
        r.register("zebra", lambda **kw: None)
        r.register("alpha", lambda **kw: None)
        r.register("middle", lambda **kw: None)
        assert r.list_plugins() == ["alpha", "middle", "zebra"]

    def test_list_plugins_empty(self):
        r = PluginRegistry("test")
        assert r.list_plugins() == []

    def test_create_passes_kwargs(self):
        def factory(*, label="default", count=0):
            return {"label": label, "count": count}
        r = PluginRegistry("test")
        r.register("myfactory", factory)
        result = r.create("myfactory", label="custom", count=42)
        assert result == {"label": "custom", "count": 42}

    def test_create_no_kwargs(self):
        r = PluginRegistry("test")
        r.register("simple", lambda: "created")
        result = r.create("simple")
        assert result == "created"

    def test_factory_exception_propagates(self):
        def bad_factory(**kw):
            raise RuntimeError("factory failed")
        r = PluginRegistry("test")
        r.register("bad", bad_factory)
        with pytest.raises(RuntimeError, match="factory failed"):
            r.create("bad")

    def test_thread_safety(self):
        r = PluginRegistry("concurrent")
        errors = []

        def register_range(start, end):
            for i in range(start, end):
                try:
                    r.register(f"plugin_{i}", lambda **kw: i)
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=register_range, args=(i * 100, (i + 1) * 100))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(r) == 500

    def test_plugin_error_hierarchy(self):
        assert issubclass(DuplicatePluginError, PluginError)
        assert issubclass(UnknownPluginError, PluginError)
        assert issubclass(PluginConfigError, PluginError)
        assert issubclass(PluginError, Exception)


# ============================================================================
# PluginRegistrySet
# ============================================================================


class TestPluginRegistrySet:
    def test_empty_init(self):
        prs = PluginRegistrySet()
        assert prs.list_domains() == []

    def test_get_registry_creates(self):
        prs = PluginRegistrySet()
        r = prs.get_registry("evaluators")
        assert isinstance(r, PluginRegistry)
        assert r.domain == "evaluators"

    def test_get_registry_returns_same(self):
        prs = PluginRegistrySet()
        r1 = prs.get_registry("evaluators")
        r2 = prs.get_registry("evaluators")
        assert r1 is r2

    def test_register_domain_explicit(self):
        prs = PluginRegistrySet()
        r = prs.register_domain("evaluators")
        assert isinstance(r, PluginRegistry)
        assert r.domain == "evaluators"

    def test_register_domain_duplicate_raises(self):
        prs = PluginRegistrySet()
        prs.register_domain("evaluators")
        with pytest.raises(DuplicatePluginError):
            prs.register_domain("evaluators")

    def test_list_domains_sorted(self):
        prs = PluginRegistrySet()
        prs.get_registry("zebra")
        prs.get_registry("alpha")
        assert prs.list_domains() == ["alpha", "zebra"]

    def test_with_defaults_domains(self):
        prs = PluginRegistrySet.with_defaults()
        domains = prs.list_domains()
        assert "evaluators" in domains
        assert "fine_tune_backends" in domains
        assert "providers" in domains
        assert "local_backends" in domains
        assert "decay_functions" in domains
        assert "middleware" in domains

    def test_with_defaults_evaluators(self):
        prs = PluginRegistrySet.with_defaults()
        r = prs.get_registry("evaluators")
        assert "exact_match" in r
        assert "structured_match" in r
        assert "semantic_similarity" in r
        assert "custom" in r
        assert len(r) == 4

    def test_with_defaults_fine_tune_backends(self):
        prs = PluginRegistrySet.with_defaults()
        r = prs.get_registry("fine_tune_backends")
        assert "unsloth" in r
        assert "openai" in r
        assert "huggingface" in r
        assert len(r) == 3

    def test_with_defaults_providers(self):
        prs = PluginRegistrySet.with_defaults()
        r = prs.get_registry("providers")
        assert "anthropic" in r
        assert "openai" in r
        assert "google" in r

    def test_with_defaults_local_backends(self):
        prs = PluginRegistrySet.with_defaults()
        r = prs.get_registry("local_backends")
        assert "ollama" in r
        assert "vllm" in r
        assert "llamacpp" in r

    def test_with_defaults_decay_functions(self):
        prs = PluginRegistrySet.with_defaults()
        r = prs.get_registry("decay_functions")
        assert "exponential" in r
        assert "linear" in r
        assert "step" in r

    def test_with_defaults_middleware_empty(self):
        prs = PluginRegistrySet.with_defaults()
        r = prs.get_registry("middleware")
        assert len(r) == 0

    def test_with_defaults_factory_creates_namespace(self):
        prs = PluginRegistrySet.with_defaults()
        r = prs.get_registry("evaluators")
        result = r.create("exact_match", threshold=0.9)
        assert result.plugin_name == "exact_match"
        assert result.threshold == 0.9

    def test_with_defaults_returns_fresh_instance(self):
        prs1 = PluginRegistrySet.with_defaults()
        prs2 = PluginRegistrySet.with_defaults()
        assert prs1 is not prs2
        # Mutating one doesn't affect the other
        prs1.get_registry("evaluators").register("extra", lambda **kw: None)
        assert "extra" not in prs2.get_registry("evaluators")

    def test_register_from_config_valid(self):
        prs = PluginRegistrySet()
        config = {
            "test_domain": {
                "my_plugin": {
                    "factory": "types.SimpleNamespace"
                }
            }
        }
        prs.register_from_config(config)
        r = prs.get_registry("test_domain")
        assert "my_plugin" in r
        result = r.create("my_plugin")
        assert isinstance(result, SimpleNamespace)

    def test_register_from_config_missing_factory_key(self):
        prs = PluginRegistrySet()
        config = {"domain": {"plugin": {"not_factory": "path"}}}
        with pytest.raises(PluginConfigError, match="Missing 'factory'"):
            prs.register_from_config(config)

    def test_register_from_config_bad_import(self):
        prs = PluginRegistrySet()
        config = {"domain": {"plugin": {"factory": "nonexistent.module.Class"}}}
        with pytest.raises(PluginConfigError, match="Failed to import"):
            prs.register_from_config(config)

    def test_register_from_config_not_callable(self):
        prs = PluginRegistrySet()
        # math.pi is a float, not callable
        config = {"domain": {"plugin": {"factory": "math.pi"}}}
        with pytest.raises(PluginConfigError, match="not callable"):
            prs.register_from_config(config)

    def test_register_from_config_bad_plugin_dict(self):
        prs = PluginRegistrySet()
        config = {"domain": {"plugin": "not_a_dict"}}
        with pytest.raises(PluginConfigError, match="Expected dict"):
            prs.register_from_config(config)

    def test_register_from_config_bad_domain_dict(self):
        prs = PluginRegistrySet()
        config = {"domain": "not_a_dict"}
        with pytest.raises(PluginConfigError, match="Expected dict"):
            prs.register_from_config(config)

    def test_register_from_config_factory_not_string(self):
        prs = PluginRegistrySet()
        config = {"domain": {"plugin": {"factory": 42}}}
        with pytest.raises(PluginConfigError, match="must be a string"):
            prs.register_from_config(config)

    def test_register_from_config_extends_defaults(self):
        prs = PluginRegistrySet.with_defaults()
        config = {
            "evaluators": {
                "my_custom_eval": {
                    "factory": "types.SimpleNamespace"
                }
            }
        }
        prs.register_from_config(config)
        r = prs.get_registry("evaluators")
        assert "my_custom_eval" in r
        # Built-in still present
        assert "exact_match" in r

    def test_register_from_config_empty(self):
        prs = PluginRegistrySet()
        prs.register_from_config({})
        assert prs.list_domains() == []


# ============================================================================
# _make_default_factory
# ============================================================================


class TestMakeDefaultFactory:
    def test_returns_callable(self):
        f = _make_default_factory("test")
        assert callable(f)

    def test_creates_namespace_with_name(self):
        f = _make_default_factory("my_plugin")
        result = f()
        assert result.plugin_name == "my_plugin"

    def test_passes_kwargs(self):
        f = _make_default_factory("test")
        result = f(a=1, b="two")
        assert result.a == 1
        assert result.b == "two"
        assert result.plugin_name == "test"


# ============================================================================
# Error attributes
# ============================================================================


class TestErrors:
    def test_duplicate_plugin_error_attrs(self):
        e = DuplicatePluginError("name", "domain")
        assert e.name == "name"
        assert e.domain == "domain"
        assert "name" in str(e)
        assert "domain" in str(e)

    def test_unknown_plugin_error_attrs(self):
        e = UnknownPluginError("name", "domain", ["a", "b"])
        assert e.name == "name"
        assert e.domain == "domain"
        assert e.available == ["a", "b"]
        assert "name" in str(e)

    def test_plugin_config_error_attrs(self):
        e = PluginConfigError("domain", "name", "bad import")
        assert e.domain == "domain"
        assert e.name == "name"
        assert e.reason == "bad import"
        assert "bad import" in str(e)

    def test_plugin_error_base(self):
        e = PluginError("generic error")
        assert str(e) == "generic error"
