"""
Middleware Pipeline — pre/post processing hooks for the Apprentice request lifecycle.

Provides a Protocol-based middleware abstraction with an onion-model pipeline:
  - pre_process runs in registration order (first-registered runs first)
  - post_process runs in reverse order (first-registered runs last — onion/LIFO)

Error in any single middleware is logged and skipped; it never prevents
other middleware from executing.

All models use frozen=True. State flows between pre and post phases via
the middleware_state dict on MiddlewareContext and MiddlewareResponse.
New instances are created (via model_copy) at each pipeline step.
"""

import logging
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)


# ===========================================================================
# Data Models
# ===========================================================================


class MiddlewareContext(BaseModel):
    """Frozen Pydantic model carrying request data through the middleware pipeline.

    Attributes:
        request_id: Unique identifier for this request.
        task_name: Name of the task being processed.
        input_data: The raw input payload.
        metadata: Arbitrary metadata associated with the request.
        middleware_state: Opaque dict for passing data from pre_process to post_process.
            Each middleware may add keys; accumulated across the pipeline via model_copy.
    """
    model_config = ConfigDict(frozen=True)

    request_id: str
    task_name: str
    input_data: dict
    metadata: dict = Field(default_factory=dict)
    middleware_state: dict = Field(default_factory=dict)


class MiddlewareResponse(BaseModel):
    """Frozen Pydantic model carrying response data through the middleware pipeline.

    Attributes:
        output_data: The output payload.
        metadata: Arbitrary metadata associated with the response.
        middleware_state: Opaque dict for passing data through the post-processing phase.
    """
    model_config = ConfigDict(frozen=True)

    output_data: dict
    metadata: dict = Field(default_factory=dict)
    middleware_state: dict = Field(default_factory=dict)


# ===========================================================================
# Errors
# ===========================================================================


class MiddlewareError(Exception):
    """Base error for middleware failures.

    Attributes:
        middleware_name: Name/class of the middleware that failed.
        phase: Which phase failed — ``"pre_process"`` or ``"post_process"``.
        reason: Human-readable description of the failure.
    """

    def __init__(self, middleware_name: str, phase: str, reason: str) -> None:
        self.middleware_name = middleware_name
        self.phase = phase
        self.reason = reason
        super().__init__(
            f"MiddlewareError in '{middleware_name}' during {phase}: {reason}"
        )


# ===========================================================================
# Protocol
# ===========================================================================


@runtime_checkable
class Middleware(Protocol):
    """Runtime-checkable protocol that all middleware implementations must satisfy."""

    def pre_process(self, context: MiddlewareContext) -> MiddlewareContext:
        """Transform the context before the core task executes.

        Must return a (potentially new) MiddlewareContext instance.
        """
        ...

    def post_process(
        self, context: MiddlewareContext, response: MiddlewareResponse
    ) -> MiddlewareResponse:
        """Transform the response after the core task executes.

        Must return a (potentially new) MiddlewareResponse instance.
        """
        ...


# ===========================================================================
# Pipeline
# ===========================================================================


class MiddlewarePipeline:
    """Ordered pipeline of Middleware instances.

    - ``execute_pre`` runs each middleware's ``pre_process`` in registration order.
    - ``execute_post`` runs each middleware's ``post_process`` in **reverse** order
      (onion/LIFO model).
    - An error in any single middleware is logged and skipped — it does **not**
      prevent the remaining middleware from running.
    - An empty pipeline is a safe no-op passthrough.
    """

    def __init__(self, middlewares: list[Middleware] | None = None) -> None:
        self._middlewares: list[Middleware] = list(middlewares) if middlewares else []

    @property
    def middlewares(self) -> list[Middleware]:
        """Return a shallow copy of the registered middleware list."""
        return list(self._middlewares)

    # ---- Pre-processing (forward order) ------------------------------------

    def execute_pre(self, context: MiddlewareContext) -> MiddlewareContext:
        """Run ``pre_process`` on each middleware in registration order.

        Middleware state is accumulated: the ``middleware_state`` dict from each
        step is merged into the next context via ``model_copy``.

        If a middleware raises, the error is logged and that middleware is
        skipped; processing continues with the remaining middleware.
        """
        current = context
        for mw in self._middlewares:
            mw_name = type(mw).__name__
            try:
                result = mw.pre_process(current)
                # Merge middleware_state from result into the running state
                merged_state = {**current.middleware_state, **result.middleware_state}
                current = result.model_copy(update={"middleware_state": merged_state})
            except Exception as exc:
                logger.error(
                    "Middleware '%s' failed during pre_process: %s",
                    mw_name,
                    exc,
                )
        return current

    # ---- Post-processing (reverse order) -----------------------------------

    def execute_post(
        self, context: MiddlewareContext, response: MiddlewareResponse
    ) -> MiddlewareResponse:
        """Run ``post_process`` on each middleware in **reverse** registration order.

        If a middleware raises, the error is logged and that middleware is
        skipped; processing continues with the remaining middleware.
        """
        current = response
        for mw in reversed(self._middlewares):
            mw_name = type(mw).__name__
            try:
                result = mw.post_process(context, current)
                # Merge middleware_state from result into the running state
                merged_state = {**current.middleware_state, **result.middleware_state}
                current = result.model_copy(update={"middleware_state": merged_state})
            except Exception as exc:
                logger.error(
                    "Middleware '%s' failed during post_process: %s",
                    mw_name,
                    exc,
                )
        return current

    # ---- Factory -----------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config_list: list[dict],
        registry: Any,
    ) -> "MiddlewarePipeline":
        """Build a pipeline from a YAML-style config list using a registry.

        Expected config format::

            [
                {"name": "pii_tokenizer", "config": {"key": "value"}},
                {"name": "rate_limiter", "config": {}},
            ]

        The *registry* must support ``registry.create(name, **config)`` which
        returns a ``Middleware``-compatible instance.

        If ``config_list`` is empty or ``None``, returns an empty (no-op) pipeline.
        """
        if not config_list:
            return cls([])

        middlewares: list[Middleware] = []
        for entry in config_list:
            name = entry.get("name", "")
            config = entry.get("config", {})
            try:
                mw = registry.create(name, **config)
                middlewares.append(mw)
            except Exception as exc:
                logger.error(
                    "Failed to create middleware '%s' from config: %s",
                    name,
                    exc,
                )
        return cls(middlewares)
