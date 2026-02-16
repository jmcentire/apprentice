"""
Composition Root — initialization, shutdown, orchestration, and backend registry.

Wires all 9 components together following a strict initialization DAG.
Teardown occurs in reverse order. No global mutable state is exposed publicly.
"""

import logging
import uuid
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

import yaml

logger = logging.getLogger("apprentice.root")


# ============================================================================
# Enums
# ============================================================================


class InitializationPhase(str, Enum):
    CONFIG_PARSE = "CONFIG_PARSE"
    REGISTRY_BUILD = "REGISTRY_BUILD"
    BUDGET_MANAGER_INIT = "BUDGET_MANAGER_INIT"
    CONFIDENCE_ENGINE_INIT = "CONFIDENCE_ENGINE_INIT"
    EXTERNAL_INTERFACES_INIT = "EXTERNAL_INTERFACES_INIT"
    SAMPLING_SCHEDULER_INIT = "SAMPLING_SCHEDULER_INIT"
    ROUTER_INIT = "ROUTER_INIT"
    TRAINING_PIPELINE_INIT = "TRAINING_PIPELINE_INIT"
    REPORTING_INIT = "REPORTING_INIT"
    UNIFIED_INTERFACE_INIT = "UNIFIED_INTERFACE_INIT"


class ComponentStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    FAILED = "FAILED"
    SHUTTING_DOWN = "SHUTTING_DOWN"
    SHUTDOWN = "SHUTDOWN"


class ComponentId(str, Enum):
    CONFIG_AND_REGISTRY = "CONFIG_AND_REGISTRY"
    BUDGET_MANAGER = "BUDGET_MANAGER"
    CONFIDENCE_ENGINE = "CONFIDENCE_ENGINE"
    EXTERNAL_INTERFACES = "EXTERNAL_INTERFACES"
    SAMPLING_SCHEDULER = "SAMPLING_SCHEDULER"
    ROUTER = "ROUTER"
    TRAINING_PIPELINE = "TRAINING_PIPELINE"
    REPORTING = "REPORTING"
    UNIFIED_INTERFACE = "UNIFIED_INTERFACE"


class FinetuningBackendName(str, Enum):
    OPENAI = "openai"
    ANYSCALE = "anyscale"
    LOCAL_LORA = "local_lora"
    AXOLOTL = "axolotl"
    CUSTOM = "custom"


class LifecycleEvent(str, Enum):
    INITIALIZATION_STARTED = "INITIALIZATION_STARTED"
    COMPONENT_INITIALIZED = "COMPONENT_INITIALIZED"
    COMPONENT_INIT_FAILED = "COMPONENT_INIT_FAILED"
    ALL_COMPONENTS_READY = "ALL_COMPONENTS_READY"
    SHUTDOWN_STARTED = "SHUTDOWN_STARTED"
    COMPONENT_SHUTDOWN = "COMPONENT_SHUTDOWN"
    COMPONENT_SHUTDOWN_FAILED = "COMPONENT_SHUTDOWN_FAILED"
    SHUTDOWN_COMPLETE = "SHUTDOWN_COMPLETE"


# ============================================================================
# Data classes
# ============================================================================


class RequestCorrelation:
    """Correlation context threaded through request processing."""

    __slots__ = ("request_id", "task_name", "created_at_utc")

    def __init__(self, request_id: str, task_name: str, created_at_utc: str):
        self.request_id = request_id
        self.task_name = task_name
        self.created_at_utc = created_at_utc


class ComponentStatusEntry:
    """Status of a single component."""

    __slots__ = ("component_id", "status", "initialization_phase", "error_message", "initialized_at_utc")

    def __init__(
        self,
        component_id: str,
        status: ComponentStatus,
        initialization_phase: str = "",
        error_message: str = "",
        initialized_at_utc: str = "",
    ):
        self.component_id = component_id
        self.status = status
        self.initialization_phase = initialization_phase
        self.error_message = error_message
        self.initialized_at_utc = initialized_at_utc


class BackendRegistryEntry:
    """A single entry in the fine-tuning backend plugin registry."""

    __slots__ = ("backend_name", "implementation", "is_remote")

    def __init__(self, backend_name: str, implementation: Any, is_remote: bool):
        self.backend_name = backend_name
        self.implementation = implementation
        self.is_remote = is_remote


class BackendRegistry:
    """Immutable dict-based registry mapping backend names to implementations."""

    __slots__ = ("entries", "active_backend_name")

    def __init__(self, entries: dict, active_backend_name: str):
        self.entries = entries
        self.active_backend_name = active_backend_name


# ============================================================================
# Errors
# ============================================================================


class AllSourcesUnavailableError(Exception):
    """Raised when both local model and remote API budget are unavailable."""

    def __init__(
        self,
        message: str,
        task_name: str,
        request_id: str,
        local_unavailable_reason: str,
        budget_exhausted_detail: str,
        budget_limit_usd: float,
        budget_used_usd: float,
        timestamp_utc: str,
    ):
        super().__init__(message)
        self.message = message
        self.task_name = task_name
        self.request_id = request_id
        self.local_unavailable_reason = local_unavailable_reason
        self.budget_exhausted_detail = budget_exhausted_detail
        self.budget_limit_usd = budget_limit_usd
        self.budget_used_usd = budget_used_usd
        self.timestamp_utc = timestamp_utc


class InitializationError(Exception):
    """Raised when the composition root fails to initialize."""

    def __init__(
        self,
        message: str,
        failed_phase: InitializationPhase,
        failed_component_id: str,
        component_statuses: list,
        cause: str,
    ):
        super().__init__(message)
        self.message = message
        self.failed_phase = failed_phase
        self.failed_component_id = failed_component_id
        self.component_statuses = component_statuses
        self.cause = cause


class ShutdownError(Exception):
    """Returned/logged when one or more components fail during shutdown."""

    def __init__(
        self,
        message: str,
        component_errors: dict,
        components_shutdown_successfully: list,
    ):
        super().__init__(message)
        self.message = message
        self.component_errors = component_errors
        self.components_shutdown_successfully = components_shutdown_successfully


# ============================================================================
# Component ID mapping
# ============================================================================

_COMPONENT_IDS = [
    "config_and_registry",
    "budget_manager",
    "confidence_engine",
    "external_interfaces",
    "sampling_scheduler",
    "router",
    "training_pipeline",
    "reporting",
    "unified_interface",
]

_COMPONENT_ID_TO_ENUM = {
    "config_and_registry": ComponentId.CONFIG_AND_REGISTRY,
    "budget_manager": ComponentId.BUDGET_MANAGER,
    "confidence_engine": ComponentId.CONFIDENCE_ENGINE,
    "external_interfaces": ComponentId.EXTERNAL_INTERFACES,
    "sampling_scheduler": ComponentId.SAMPLING_SCHEDULER,
    "router": ComponentId.ROUTER,
    "training_pipeline": ComponentId.TRAINING_PIPELINE,
    "reporting": ComponentId.REPORTING,
    "unified_interface": ComponentId.UNIFIED_INTERFACE,
}

_COMPONENT_PHASE = {
    "config_and_registry": InitializationPhase.CONFIG_PARSE,
    "budget_manager": InitializationPhase.BUDGET_MANAGER_INIT,
    "confidence_engine": InitializationPhase.CONFIDENCE_ENGINE_INIT,
    "external_interfaces": InitializationPhase.EXTERNAL_INTERFACES_INIT,
    "sampling_scheduler": InitializationPhase.SAMPLING_SCHEDULER_INIT,
    "router": InitializationPhase.ROUTER_INIT,
    "training_pipeline": InitializationPhase.TRAINING_PIPELINE_INIT,
    "reporting": InitializationPhase.REPORTING_INIT,
    "unified_interface": InitializationPhase.UNIFIED_INTERFACE_INIT,
}

# Required methods per component for override validation
_REQUIRED_METHODS = {
    "config_and_registry": ["load_config"],
    "budget_manager": ["authorize", "record_spend", "release_reservation", "record_local_request", "remaining_budget", "force_persist"],
    "confidence_engine": ["get_snapshot", "record_comparison", "persist_state", "load_state"],
    "external_interfaces": ["complete", "health", "close"],
    "sampling_scheduler": ["should_sample"],
    "router": ["route"],
    "training_pipeline": ["add_example", "should_trigger", "run_pipeline"],
    "reporting": ["audit_logger"],
    "unified_interface": ["run"],
}

# Remote backends requiring api_key and api_base_url
_REMOTE_BACKENDS = {"openai", "anyscale"}
_LOCAL_BACKENDS = {"local_lora", "axolotl"}
_BUILTIN_BACKENDS = _REMOTE_BACKENDS | _LOCAL_BACKENDS


# ============================================================================
# Placeholder factory stubs (patchable in tests)
# ============================================================================


class BudgetManager:
    """Placeholder — patched in tests."""
    pass


class ConfidenceEngine:
    """Placeholder — patched in tests."""
    pass


class Router:
    """Placeholder — patched in tests."""
    pass


class JsonLinesAuditLogger:
    """Placeholder — patched in tests."""
    pass


def create_remote_adapter(*args, **kwargs):
    """Placeholder — patched in tests."""
    pass


# ============================================================================
# Private module state (underscore-prefixed, allowed by invariant)
# ============================================================================

_components: Optional[dict] = None


# ============================================================================
# Public functions
# ============================================================================


def _parse_config(config_path: str, env: dict) -> dict:
    """Parse and validate config from YAML file, resolving env vars."""
    import os

    if not os.path.exists(config_path):
        raise InitializationError(
            message=f"Config file not found: {config_path}",
            failed_phase=InitializationPhase.CONFIG_PARSE,
            failed_component_id="CONFIG_AND_REGISTRY",
            component_statuses=[],
            cause=f"Config file not found: {config_path}",
        )

    try:
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)
    except Exception as e:
        raise InitializationError(
            message=f"Config parse error: {e}",
            failed_phase=InitializationPhase.CONFIG_PARSE,
            failed_component_id="CONFIG_AND_REGISTRY",
            component_statuses=[],
            cause=str(e),
        )

    if raw_config is None:
        raw_config = {}

    # Validate required sections
    required_sections = ["budget", "confidence_engine", "external", "finetuning", "sampling", "reporting"]
    missing = [s for s in required_sections if s not in raw_config or raw_config[s] is None]
    if missing:
        raise InitializationError(
            message=f"Config validation failed: missing required sections: {missing}",
            failed_phase=InitializationPhase.CONFIG_PARSE,
            failed_component_id="CONFIG_AND_REGISTRY",
            component_statuses=[],
            cause=f"Missing required config sections: {missing}",
        )

    # Resolve env: references
    _resolve_env_vars(raw_config, env)

    # Validate finetuning backend
    ft_backend = raw_config.get("finetuning", {}).get("backend", "")
    if ft_backend and ft_backend not in _BUILTIN_BACKENDS and ft_backend != "custom":
        raise InitializationError(
            message=f"Unsupported fine-tuning backend: {ft_backend}",
            failed_phase=InitializationPhase.CONFIG_PARSE,
            failed_component_id="CONFIG_AND_REGISTRY",
            component_statuses=[],
            cause=f"Unknown finetuning backend: {ft_backend}",
        )

    return raw_config


def _resolve_env_vars(obj, env: dict):
    """Recursively resolve env:VAR_NAME references in config."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str) and value.startswith("env:"):
                var_name = value[4:]
                if var_name not in env:
                    raise InitializationError(
                        message=f"Environment variable not found: {var_name}",
                        failed_phase=InitializationPhase.CONFIG_PARSE,
                        failed_component_id="CONFIG_AND_REGISTRY",
                        component_statuses=[],
                        cause=f"Environment variable '{var_name}' not found in env mapping",
                    )
                obj[key] = env[var_name]
            elif isinstance(value, (dict, list)):
                _resolve_env_vars(value, env)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, str) and item.startswith("env:"):
                var_name = item[4:]
                if var_name not in env:
                    raise InitializationError(
                        message=f"Environment variable not found: {var_name}",
                        failed_phase=InitializationPhase.CONFIG_PARSE,
                        failed_component_id="CONFIG_AND_REGISTRY",
                        component_statuses=[],
                        cause=f"Environment variable '{var_name}' not found in env mapping",
                    )
                obj[i] = env[var_name]
            elif isinstance(item, (dict, list)):
                _resolve_env_vars(item, env)


async def _teardown_components(initialized: dict):
    """Best-effort teardown of already-initialized components in reverse order."""
    for cid in reversed(_COMPONENT_IDS):
        comp = initialized.get(cid)
        if comp is None:
            continue
        try:
            if hasattr(comp, "force_persist") and callable(comp.force_persist):
                await comp.force_persist()
            if hasattr(comp, "persist_state") and callable(comp.persist_state):
                await comp.persist_state()
            if hasattr(comp, "close") and callable(comp.close):
                await comp.close()
        except Exception:
            pass  # Best effort during teardown on init failure


async def initialize_composition_root(
    config_path: str,
    env: dict = None,
    component_overrides: dict = None,
) -> dict:
    """Construct and wire all 9 components in DAG order."""
    global _components

    if env is None:
        import os
        env = dict(os.environ)
    if component_overrides is None:
        component_overrides = {}

    logger.info("Initialization started")

    # Phase 1: Parse config
    raw_config = _parse_config(config_path, env)

    initialized = {}

    # DAG order init phases
    _init_phases = [
        ("config_and_registry", InitializationPhase.CONFIG_PARSE),
        ("budget_manager", InitializationPhase.BUDGET_MANAGER_INIT),
        ("confidence_engine", InitializationPhase.CONFIDENCE_ENGINE_INIT),
        ("external_interfaces", InitializationPhase.EXTERNAL_INTERFACES_INIT),
        ("sampling_scheduler", InitializationPhase.SAMPLING_SCHEDULER_INIT),
        ("router", InitializationPhase.ROUTER_INIT),
        ("training_pipeline", InitializationPhase.TRAINING_PIPELINE_INIT),
        ("reporting", InitializationPhase.REPORTING_INIT),
        ("unified_interface", InitializationPhase.UNIFIED_INTERFACE_INIT),
    ]

    for cid, phase in _init_phases:
        if cid in component_overrides:
            initialized[cid] = component_overrides[cid]
            logger.debug(f"Component {cid} initialized (override)")
            continue

        try:
            component = _construct_component(cid, raw_config, initialized, env)
            initialized[cid] = component
            logger.debug(f"Component {cid} initialized")
        except InitializationError:
            logger.error(f"Component {cid} failed to initialize")
            await _teardown_components(initialized)
            raise
        except Exception as e:
            logger.error(f"Component {cid} failed to initialize: {e}")
            await _teardown_components(initialized)
            raise InitializationError(
                message=f"Failed to initialize {cid}: {e}",
                failed_phase=phase,
                failed_component_id=cid.upper(),
                component_statuses=[],
                cause=str(e),
            )

    logger.info("All components ready")
    _components = initialized
    return initialized


def _construct_component(cid: str, raw_config: dict, initialized: dict, env: dict) -> Any:
    """Construct a single component from config. Raises on failure."""
    if cid == "config_and_registry":
        # Build a simple config wrapper from raw_config
        from types import SimpleNamespace
        config = SimpleNamespace(**raw_config)
        config.budget = SimpleNamespace(**raw_config.get("budget", {}))
        config.confidence_engine = SimpleNamespace(**raw_config.get("confidence_engine", {}))
        config.external = SimpleNamespace(**raw_config.get("external", {}))
        config.finetuning = SimpleNamespace(**raw_config.get("finetuning", {}))
        config.sampling = SimpleNamespace(**raw_config.get("sampling", {}))
        config.reporting = SimpleNamespace(**raw_config.get("reporting", {}))
        config.tasks = raw_config.get("tasks", [])

        cr = SimpleNamespace()
        cr.config = config
        cr.load_config = lambda: config
        cr.task_registry = SimpleNamespace()
        cr.task_registry.get_task = lambda name: None
        cr.task_registry.__contains__ = lambda name: False
        cr.task_registry.task_names = lambda: [t.get("name", "") for t in config.tasks] if isinstance(config.tasks, list) else []
        return cr

    elif cid == "budget_manager":
        return BudgetManager()

    elif cid == "confidence_engine":
        return ConfidenceEngine()

    elif cid == "external_interfaces":
        remote_config = raw_config.get("external", {}).get("remote", {})
        return create_remote_adapter(remote_config)

    elif cid == "sampling_scheduler":
        from types import SimpleNamespace
        return SimpleNamespace(should_sample=lambda *a, **kw: None)

    elif cid == "router":
        return Router()

    elif cid == "training_pipeline":
        ft_config = raw_config.get("finetuning", {})
        backend_name = ft_config.get("backend", "")
        if backend_name not in _BUILTIN_BACKENDS and backend_name != "custom":
            raise ValueError(f"Unsupported fine-tuning backend: {backend_name}")
        from types import SimpleNamespace
        return SimpleNamespace(
            add_example=lambda *a, **kw: None,
            should_trigger=lambda *a, **kw: None,
            run_pipeline=lambda *a, **kw: None,
            get_status=lambda *a, **kw: None,
            get_run_history=lambda *a, **kw: None,
            initialize=lambda *a, **kw: None,
        )

    elif cid == "reporting":
        audit_config = raw_config.get("reporting", {})
        audit_logger = JsonLinesAuditLogger(audit_config)
        from types import SimpleNamespace
        rp = SimpleNamespace()
        rp.audit_logger = audit_logger
        rp.generate_report = lambda *a, **kw: None
        rp.format_report = lambda *a, **kw: None
        return rp

    elif cid == "unified_interface":
        from types import SimpleNamespace
        return SimpleNamespace(
            run=lambda *a, **kw: None,
            status=lambda *a, **kw: None,
            report=lambda *a, **kw: None,
            close=lambda *a, **kw: None,
        )

    raise ValueError(f"Unknown component: {cid}")


async def shutdown_composition_root(
    components: dict,
    exit_stack: Any,
) -> ShutdownError:
    """Tear down all components in reverse DAG order."""
    component_errors = {}
    components_ok = []

    logger.info("Shutdown started")

    # Reverse DAG order
    shutdown_order = list(reversed(_COMPONENT_IDS))

    for cid in shutdown_order:
        comp = components.get(cid)
        if comp is None:
            components_ok.append(cid)
            continue

        try:
            # Component-specific shutdown actions
            if cid == "unified_interface":
                if hasattr(comp, "close") and callable(comp.close):
                    await comp.close()

            elif cid == "reporting":
                if hasattr(comp, "audit_logger"):
                    al = comp.audit_logger
                    if hasattr(al, "flush") and callable(al.flush):
                        await al.flush()
                    if hasattr(al, "close") and callable(al.close):
                        await al.close()

            elif cid == "training_pipeline":
                pass  # No special shutdown

            elif cid == "router":
                pass  # No special shutdown

            elif cid == "sampling_scheduler":
                pass  # No special shutdown

            elif cid == "external_interfaces":
                if hasattr(comp, "close") and callable(comp.close):
                    await comp.close()

            elif cid == "confidence_engine":
                if hasattr(comp, "persist_state") and callable(comp.persist_state):
                    await comp.persist_state()

            elif cid == "budget_manager":
                if hasattr(comp, "force_persist") and callable(comp.force_persist):
                    await comp.force_persist()

            elif cid == "config_and_registry":
                pass  # No special shutdown

            components_ok.append(cid)
            logger.debug(f"Component {cid} shut down")

        except Exception as e:
            component_errors[cid] = str(e)
            logger.error(f"Component {cid} shutdown failed: {e}")

    # Close exit stack
    try:
        if exit_stack is not None and hasattr(exit_stack, "aclose"):
            await exit_stack.aclose()
    except Exception as e:
        component_errors["exit_stack"] = str(e)
        logger.error(f"Exit stack close failed: {e}")

    message = "Shutdown complete" if not component_errors else "Shutdown completed with errors"
    logger.info(message)

    return ShutdownError(
        message=message,
        component_errors=component_errors,
        components_shutdown_successfully=components_ok,
    )


def resolve_finetuning_backend(
    backend_name: str,
    finetuning_config: dict,
    custom_backends: dict = None,
) -> BackendRegistryEntry:
    """Resolve a fine-tuning backend plugin from the configured name."""
    if custom_backends is None:
        custom_backends = {}

    backend_name = backend_name.strip()
    if not backend_name:
        raise ValueError("Unknown fine-tuning backend: empty backend_name not allowed")

    if backend_name in _REMOTE_BACKENDS:
        if "api_key" not in finetuning_config or not finetuning_config["api_key"]:
            raise ValueError(
                f"Remote fine-tuning backend '{backend_name}' requires api_key in config"
            )
        if "api_base_url" not in finetuning_config or not finetuning_config["api_base_url"]:
            raise ValueError(
                f"Remote fine-tuning backend '{backend_name}' requires api_base_url in config"
            )
        return BackendRegistryEntry(
            backend_name=backend_name,
            implementation=_make_builtin_backend(backend_name, finetuning_config),
            is_remote=True,
        )

    if backend_name in _LOCAL_BACKENDS:
        return BackendRegistryEntry(
            backend_name=backend_name,
            implementation=_make_builtin_backend(backend_name, finetuning_config),
            is_remote=False,
        )

    if backend_name == "custom":
        if not custom_backends:
            raise ValueError(
                "Custom fine-tuning backend requested but no custom implementation registered"
            )
        # Use the first (or only) custom backend
        custom_name = next(iter(custom_backends))
        return BackendRegistryEntry(
            backend_name="custom",
            implementation=custom_backends[custom_name],
            is_remote=False,
        )

    raise ValueError(
        f"Unknown fine-tuning backend: '{backend_name}'. "
        f"Available backends: openai, anyscale, local_lora, axolotl, custom"
    )


def _make_builtin_backend(backend_name: str, config: dict) -> Any:
    """Create a stub implementation for a built-in backend."""
    from types import SimpleNamespace
    return SimpleNamespace(
        name=backend_name,
        config=config,
        train=lambda *a, **kw: None,
    )


def build_backend_registry(
    finetuning_config: dict,
    custom_backends: dict = None,
) -> BackendRegistry:
    """Construct the BackendRegistry from config."""
    if custom_backends is None:
        custom_backends = {}

    entries = {}

    # Register all built-in backends (with dummy configs for non-active ones)
    for name in ["openai", "anyscale", "local_lora", "axolotl"]:
        is_remote = name in _REMOTE_BACKENDS
        try:
            if name in _REMOTE_BACKENDS:
                # Use placeholder config for non-active remote backends
                cfg = dict(finetuning_config)
                cfg.setdefault("api_key", "placeholder")
                cfg.setdefault("api_base_url", "https://placeholder.example.com/v1")
                impl = _make_builtin_backend(name, cfg)
            else:
                impl = _make_builtin_backend(name, finetuning_config)
            entries[name] = BackendRegistryEntry(
                backend_name=name,
                implementation=impl,
                is_remote=is_remote,
            )
        except Exception:
            pass  # Skip backends that can't be constructed

    # Register custom backends
    for custom_name, custom_impl in custom_backends.items():
        entries["custom"] = BackendRegistryEntry(
            backend_name="custom",
            implementation=custom_impl,
            is_remote=False,
        )

    active = finetuning_config.get("backend", "")
    if active not in entries:
        raise ValueError(
            f"Active fine-tuning backend '{active}' is not registered. "
            f"Available: {list(entries.keys())}"
        )

    return BackendRegistry(entries=entries, active_backend_name=active)


async def orchestrate_request(
    task_name: str,
    input_data: dict,
    correlation: RequestCorrelation,
) -> dict:
    """Top-level request orchestration through all components."""
    components = _components
    if components is None:
        raise RuntimeError("Composition root not initialized")

    cr = components.get("config_and_registry")
    bm = components.get("budget_manager")
    ce = components.get("confidence_engine")
    ss = components.get("sampling_scheduler")
    router = components.get("router")
    tp = components.get("training_pipeline")
    rp = components.get("reporting")

    # Step 1: Look up task
    task_reg = getattr(cr, "task_registry", None)
    if task_reg is not None:
        contains = False
        if hasattr(task_reg, "__contains__"):
            try:
                contains = task_name in task_reg
            except Exception:
                contains = False
        if not contains:
            try:
                task_reg.get_task(task_name)
            except (KeyError, Exception) as e:
                raise ValueError(f"Task not found: {task_name}") from e

    # Step 2: Get confidence snapshot
    snapshot = await ce.get_snapshot(task_name)

    # Step 3: Get sampling decision
    sampling_decision = ss.should_sample(task_name, snapshot)

    # Step 4: Determine if remote is needed
    decision = getattr(sampling_decision, "decision", "local_only")
    needs_remote = decision in ("remote_only", "dual_send", "local_with_remote_fallback")
    budget_authorized = False
    auth_result = None

    if needs_remote:
        # Step 5: Budget authorization
        auth_result = await bm.authorize(task_name)
        budget_authorized = getattr(auth_result, "allowed", False)

        if not budget_authorized and decision == "local_only":
            pass  # Proceed with local
        elif not budget_authorized and decision in ("remote_only",):
            # Can't use remote, no budget
            pass  # Will try to route and may fail

    # Step 6: Route request
    route_result = None
    remote_failed = False
    try:
        route_result = await router.route(
            task_name=task_name,
            input_data=input_data,
            correlation=correlation,
            sampling_decision=sampling_decision,
            budget_authorized=budget_authorized,
        )
    except AllSourcesUnavailableError:
        # Log audit entry before re-raising
        try:
            await rp.audit_logger.log({
                "task_name": task_name,
                "request_id": correlation.request_id,
                "decision": decision,
                "error": "AllSourcesUnavailable",
            })
        except Exception:
            pass
        raise
    except Exception as e:
        remote_failed = True
        # Step 7: Release reservation on failure
        if needs_remote and budget_authorized:
            try:
                await bm.release_reservation(task_name)
            except Exception:
                pass
        # Log audit entry before re-raising
        try:
            await rp.audit_logger.log({
                "task_name": task_name,
                "request_id": correlation.request_id,
                "decision": decision,
                "error": str(e),
            })
        except Exception:
            pass
        raise

    # Step 7: Budget reconciliation
    used_remote = getattr(route_result, "used_remote", False)
    used_local = getattr(route_result, "used_local", False)
    remote_cost = getattr(route_result, "remote_cost", None)

    if used_remote and remote_cost is not None:
        try:
            await bm.record_spend(task_name, remote_cost)
        except Exception:
            pass

    if used_local and not used_remote:
        try:
            await bm.record_local_request(task_name)
        except Exception:
            pass

    # Step 8: Coaching sample — record comparison
    is_coaching = getattr(sampling_decision, "is_coaching_sample", False)
    if is_coaching and used_remote and used_local:
        try:
            local_output = getattr(route_result, "local_output", None)
            remote_output = getattr(route_result, "remote_output", None)
            await ce.record_comparison(task_name, local_output, remote_output)
        except Exception:
            pass

    # Step 9: Training data
    if is_coaching and used_remote:
        try:
            await tp.add_example(task_name, input_data, route_result)
        except Exception:
            pass

    # Step 10: Audit log entry
    response = getattr(route_result, "response", route_result)
    try:
        await rp.audit_logger.log({
            "task_name": task_name,
            "request_id": correlation.request_id,
            "decision": decision,
            "used_remote": used_remote,
            "used_local": used_local,
        })
    except Exception:
        pass  # Audit failure must not block response

    return response


def get_component_statuses(components: dict) -> list:
    """Return status of all 9 managed components."""
    result = []
    for cid in _COMPONENT_IDS:
        comp = components.get(cid)
        if comp is None:
            result.append(ComponentStatusEntry(
                component_id=cid.upper(),
                status=ComponentStatus.NOT_STARTED,
                initialization_phase=_COMPONENT_PHASE.get(cid, "").value if cid in _COMPONENT_PHASE else "",
            ))
        else:
            # Check if component has been marked as failed
            status_attr = getattr(comp, "_status", None)
            error_attr = getattr(comp, "_error", None)
            if status_attr == "FAILED":
                result.append(ComponentStatusEntry(
                    component_id=cid.upper(),
                    status=ComponentStatus.FAILED,
                    initialization_phase=_COMPONENT_PHASE.get(cid, "").value if cid in _COMPONENT_PHASE else "",
                    error_message=error_attr or "Unknown error",
                ))
            else:
                result.append(ComponentStatusEntry(
                    component_id=cid.upper(),
                    status=ComponentStatus.READY,
                    initialization_phase=_COMPONENT_PHASE.get(cid, "").value if cid in _COMPONENT_PHASE else "",
                ))
    return result


def create_request_correlation(task_name: str) -> RequestCorrelation:
    """Create a new RequestCorrelation with UUID v4 and UTC timestamp."""
    if not task_name or not task_name.strip():
        raise ValueError("task_name must be a non-empty, non-whitespace string")

    return RequestCorrelation(
        request_id=str(uuid.uuid4()),
        task_name=task_name,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def validate_component_override(component_id: str, override_instance: Any) -> bool:
    """Validate that an override conforms to the expected Protocol."""
    if component_id not in _REQUIRED_METHODS:
        raise ValueError(f"Unknown component_id: '{component_id}'. Valid IDs: {list(_REQUIRED_METHODS.keys())}")

    required = _REQUIRED_METHODS[component_id]
    for method_name in required:
        if not hasattr(override_instance, method_name):
            return False
    return True
