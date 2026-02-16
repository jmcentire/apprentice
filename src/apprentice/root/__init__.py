"""
Root composition package.

Re-exports all public API from composition module.
"""

from src.root.composition import (
    initialize_composition_root,
    shutdown_composition_root,
    resolve_finetuning_backend,
    orchestrate_request,
    get_component_statuses,
    create_request_correlation,
    validate_component_override,
    build_backend_registry,
    AllSourcesUnavailableError,
    InitializationError,
    ShutdownError,
    RequestCorrelation,
    ComponentId,
    ComponentStatus,
    InitializationPhase,
    LifecycleEvent,
    FinetuningBackendName,
    BackendRegistryEntry,
    BackendRegistry,
)

__all__ = [
    "initialize_composition_root",
    "shutdown_composition_root",
    "resolve_finetuning_backend",
    "orchestrate_request",
    "get_component_statuses",
    "create_request_correlation",
    "validate_component_override",
    "build_backend_registry",
    "AllSourcesUnavailableError",
    "InitializationError",
    "ShutdownError",
    "RequestCorrelation",
    "ComponentId",
    "ComponentStatus",
    "InitializationPhase",
    "LifecycleEvent",
    "FinetuningBackendName",
    "BackendRegistryEntry",
    "BackendRegistry",
]
