"""
Plugin Registry — instance-based, domain-scoped plugin registration system.

Replaces hardcoded enums with string-keyed factory registries.
Thread-safe, O(1) lookup, no global state.
"""

import importlib
import re
import threading
from types import SimpleNamespace
from typing import Any, Callable


# ============================================================================
# Errors
# ============================================================================


class PluginError(Exception):
    """Base error for all plugin registry errors."""


class DuplicatePluginError(PluginError):
    """Raised when registering a name that already exists."""

    def __init__(self, name: str, domain: str):
        self.name = name
        self.domain = domain
        super().__init__(f"Plugin '{name}' already registered in domain '{domain}'")


class UnknownPluginError(PluginError):
    """Raised when looking up a name that doesn't exist."""

    def __init__(self, name: str, domain: str, available: list[str]):
        self.name = name
        self.domain = domain
        self.available = available
        super().__init__(
            f"Unknown plugin '{name}' in domain '{domain}'. "
            f"Available: {available}"
        )


class PluginConfigError(PluginError):
    """Raised when config-based registration fails."""

    def __init__(self, domain: str, name: str, reason: str):
        self.domain = domain
        self.name = name
        self.reason = reason
        super().__init__(
            f"Failed to register plugin '{name}' in domain '{domain}': {reason}"
        )


# ============================================================================
# Name validation
# ============================================================================

_PLUGIN_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate_plugin_name(name: str) -> None:
    """Validate that a plugin name matches the required pattern."""
    if not name:
        raise ValueError("Plugin name must be non-empty")
    if not _PLUGIN_NAME_RE.match(name):
        raise ValueError(
            f"Plugin name '{name}' must match ^[a-z][a-z0-9_]*$ "
            f"(lowercase letters, digits, underscores, starting with a letter)"
        )


# ============================================================================
# PluginRegistry
# ============================================================================


class PluginRegistry:
    """Instance-based, domain-scoped plugin registry.

    Maps string names to factory callables. O(1) lookup. Thread-safe.
    """

    def __init__(self, domain: str):
        if not domain or not domain.strip():
            raise ValueError("Domain must be a non-empty string")
        self._domain = domain
        self._factories: dict[str, Callable[..., Any]] = {}
        self._lock = threading.Lock()

    @property
    def domain(self) -> str:
        return self._domain

    def register(self, name: str, factory: Callable[..., Any]) -> None:
        """Register a factory callable under the given name."""
        _validate_plugin_name(name)
        if not callable(factory):
            raise TypeError(f"Factory must be callable, got {type(factory).__name__}")
        with self._lock:
            if name in self._factories:
                raise DuplicatePluginError(name, self._domain)
            self._factories[name] = factory

    def create(self, name: str, **config: Any) -> Any:
        """Look up factory by name, call it with **config, return the result."""
        with self._lock:
            factory = self._factories.get(name)
        if factory is None:
            raise UnknownPluginError(name, self._domain, self.list_plugins())
        return factory(**config)

    def validate_name(self, name: str) -> bool:
        """Return True if name is registered, False otherwise."""
        with self._lock:
            return name in self._factories

    def list_plugins(self) -> list[str]:
        """Return sorted list of registered plugin names."""
        with self._lock:
            return sorted(self._factories.keys())

    def __contains__(self, name: str) -> bool:
        return self.validate_name(name)

    def __len__(self) -> int:
        with self._lock:
            return len(self._factories)


# ============================================================================
# PluginRegistrySet
# ============================================================================


def _make_default_factory(name: str) -> Callable[..., Any]:
    """Create a placeholder factory that returns a SimpleNamespace with config."""
    def factory(**config: Any) -> SimpleNamespace:
        return SimpleNamespace(plugin_name=name, **config)
    return factory


class PluginRegistrySet:
    """Bundles multiple PluginRegistry instances, one per domain."""

    def __init__(self):
        self._registries: dict[str, PluginRegistry] = {}
        self._lock = threading.Lock()

    def get_registry(self, domain: str) -> PluginRegistry:
        """Return the registry for the given domain, creating if needed."""
        with self._lock:
            if domain not in self._registries:
                self._registries[domain] = PluginRegistry(domain)
            return self._registries[domain]

    def register_domain(self, domain: str) -> PluginRegistry:
        """Explicitly create and return a new registry for the domain."""
        with self._lock:
            if domain in self._registries:
                raise DuplicatePluginError(domain, "domains")
            registry = PluginRegistry(domain)
            self._registries[domain] = registry
            return registry

    @classmethod
    def with_defaults(cls) -> "PluginRegistrySet":
        """Create a new PluginRegistrySet pre-populated with built-in registries."""
        registry_set = cls()

        # Evaluators
        evaluators = registry_set.get_registry("evaluators")
        for name in ("exact_match", "structured_match", "semantic_similarity", "custom"):
            evaluators.register(name, _make_default_factory(name))

        # Fine-tune backends
        ft_backends = registry_set.get_registry("fine_tune_backends")
        for name in ("unsloth", "openai", "huggingface"):
            ft_backends.register(name, _make_default_factory(name))

        # Providers
        providers = registry_set.get_registry("providers")
        for name in ("anthropic", "openai", "google"):
            providers.register(name, _make_default_factory(name))

        # Local backends
        local_backends = registry_set.get_registry("local_backends")
        for name in ("ollama", "vllm", "llamacpp"):
            local_backends.register(name, _make_default_factory(name))

        # Decay functions
        decay_functions = registry_set.get_registry("decay_functions")
        for name in ("exponential", "linear", "step"):
            decay_functions.register(name, _make_default_factory(name))

        # Middleware (empty)
        registry_set.get_registry("middleware")

        return registry_set

    def register_from_config(self, config_dict: dict) -> None:
        """Register plugins from a config dict.

        Format: {"domain": {"name": {"factory": "dotted.path.ClassName"}, ...}, ...}
        """
        for domain, plugins in config_dict.items():
            if not isinstance(plugins, dict):
                raise PluginConfigError(domain, "", f"Expected dict, got {type(plugins).__name__}")
            registry = self.get_registry(domain)
            for name, plugin_config in plugins.items():
                if not isinstance(plugin_config, dict):
                    raise PluginConfigError(domain, name, f"Expected dict, got {type(plugin_config).__name__}")
                factory_path = plugin_config.get("factory")
                if not factory_path:
                    raise PluginConfigError(domain, name, "Missing 'factory' key")
                if not isinstance(factory_path, str):
                    raise PluginConfigError(domain, name, f"'factory' must be a string, got {type(factory_path).__name__}")
                try:
                    module_path, attr_name = factory_path.rsplit(".", 1)
                    module = importlib.import_module(module_path)
                    factory = getattr(module, attr_name)
                except (ValueError, ImportError, AttributeError) as e:
                    raise PluginConfigError(domain, name, f"Failed to import '{factory_path}': {e}")
                if not callable(factory):
                    raise PluginConfigError(domain, name, f"'{factory_path}' is not callable")
                registry.register(name, factory)

    def list_domains(self) -> list[str]:
        """Return sorted list of domain names."""
        with self._lock:
            return sorted(self._registries.keys())
