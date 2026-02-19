"""
WOS HTTP API — exposes the WOS Copilot over HTTP using Starlette.

Endpoints:
  POST /api/v1/events       — Ingest WOS event (fire-and-forget, 200 OK always)
  POST /api/v1/recommend    — Request recommendation (sync, ≤5s)
  POST /api/v1/feedback     — Submit agent feedback
  GET  /api/v1/status/{skill} — Skill confidence/phase
  GET  /api/v1/skills       — List skills with modes
  GET  /api/v1/health       — Health check (no auth)
"""

import logging
import time
import uuid
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from starlette.applications import Starlette
from starlette.middleware import Middleware as StarletteMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from apprentice.feedback_collector import FeedbackCollector, FeedbackEntry, FeedbackType
from apprentice.middleware import MiddlewarePipeline
from apprentice.observer import Observer, ObserverConfig
from apprentice.wos_event_adapter import (
    WOSContextBuilder,
    WOSEventAdapter,
    validate_wos_event,
)
from apprentice.wos_pii_patterns import create_wos_pii_tokenizer
from apprentice.wos_recommendation_engine import (
    RecommendationEngine,
    create_recommendation_engine,
)
from apprentice.wos_skill_definitions import create_skill_registry

logger = logging.getLogger(__name__)


# ===========================================================================
# Config
# ===========================================================================


class WOSCopilotConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    api_key: str
    rate_limit_per_minute: int = Field(default=60, ge=1)
    observer_enabled: bool = True
    observer_context_window: int = Field(default=50, ge=1)
    observer_shadow_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    feedback_enabled: bool = True
    feedback_storage_dir: str = ".apprentice/feedback/"
    version: str = "0.1.0"


# ===========================================================================
# Rate Limiter
# ===========================================================================


class RateLimiter:
    """Sliding-window rate limiter keyed by organization_id."""

    def __init__(self, max_requests_per_minute: int = 60) -> None:
        self._max_rpm = max_requests_per_minute
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, organization_id: str) -> bool:
        now = time.monotonic()
        window_start = now - 60.0
        timestamps = self._requests[organization_id]
        # Prune old entries
        self._requests[organization_id] = [t for t in timestamps if t > window_start]
        if len(self._requests[organization_id]) >= self._max_rpm:
            return False
        self._requests[organization_id].append(now)
        return True


# ===========================================================================
# Auth
# ===========================================================================


class APIKeyAuth:
    """Validates X-Apprentice-Key header."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def is_valid(self, request: Request) -> bool:
        key = request.headers.get("x-apprentice-key", "")
        return key == self._api_key


# ===========================================================================
# WOS Copilot Service (orchestrator)
# ===========================================================================


class WOSCopilotService:
    """Orchestrates all Apprentice components for the WOS copilot."""

    def __init__(self, config: WOSCopilotConfig) -> None:
        self._config = config

        # PII tokenizer as middleware
        pii_tokenizer = create_wos_pii_tokenizer()
        pipeline = MiddlewarePipeline(middlewares=[pii_tokenizer])

        # Observer
        observer_config = ObserverConfig(
            enabled=config.observer_enabled,
            context_window_size=config.observer_context_window,
            shadow_recommendation_rate=config.observer_shadow_rate,
        )
        self._observer = Observer(observer_config)

        # Event adapter + context builder
        self._event_adapter = WOSEventAdapter()
        self._context_builder = WOSContextBuilder()

        # Recommendation engine
        self._engine = create_recommendation_engine(middleware_pipeline=pipeline)

        # Feedback collector
        self._feedback = FeedbackCollector(
            storage_dir=config.feedback_storage_dir,
            enabled=config.feedback_enabled,
        )

        # Skill registry for status lookups
        self._registry = create_skill_registry()

    def handle_event(self, event_data: dict) -> dict:
        """Process incoming WOS event. Never raises — returns status dict."""
        try:
            wos_event = validate_wos_event(event_data)
            obs_event = self._event_adapter.adapt(wos_event)
            self._observer.observe(obs_event)
            self._context_builder.add_event(wos_event)
            return {"status": "accepted"}
        except Exception as exc:
            logger.warning("Event processing error: %s", exc)
            return {"status": "accepted"}

    def handle_recommend(
        self, skill: str, context: dict, request_id: str | None = None
    ) -> dict:
        """Generate recommendation. Raises KeyError/ValueError on bad input."""
        rec = self._engine.recommend(skill, context, request_id=request_id)
        return rec.model_dump()

    def handle_feedback(self, feedback_data: dict) -> dict:
        """Record feedback. Raises ValueError/KeyError on bad input."""
        request_id = feedback_data.get("request_id")
        skill = feedback_data.get("skill")
        feedback_type_str = feedback_data.get("feedback_type")

        if not request_id or not skill or not feedback_type_str:
            raise ValueError("Missing required fields: request_id, skill, feedback_type")

        try:
            feedback_type = FeedbackType(feedback_type_str)
        except ValueError:
            raise ValueError(
                f"Invalid feedback_type '{feedback_type_str}'. "
                f"Must be one of: {[ft.value for ft in FeedbackType]}"
            )

        entry = FeedbackEntry(
            request_id=request_id,
            task_name=skill,
            feedback_type=feedback_type,
            edited_output=feedback_data.get("edited_output"),
            reason=feedback_data.get("reason"),
        )
        self._feedback.record_feedback(entry)
        return {"status": "recorded"}

    def get_skill_status(self, skill: str) -> dict:
        """Get skill status. Raises KeyError if unknown."""
        config = self._registry.get_skill(skill)
        summary = self._feedback.get_feedback_summary(skill)
        context = self._observer.get_context(skill)
        return {
            "skill": skill,
            "confidence": summary.acceptance_rate,
            "phase": "observer",
            "mode": "copilot",
            "event_count": len(context),
            "feedback_count": summary.total_count,
            "risk_level": config.risk_level.value,
        }

    def list_skills(self) -> list[dict]:
        skills = self._registry.list_skills()
        result = []
        for name in skills:
            config = self._registry.get_skill(name)
            summary = self._feedback.get_feedback_summary(name)
            result.append({
                "name": name,
                "risk_level": config.risk_level.value,
                "mode": "copilot",
                "confidence": summary.acceptance_rate,
            })
        return result

    def health(self) -> dict:
        return {
            "status": "healthy",
            "version": self._config.version,
            "skills_loaded": len(self._registry.list_skills()),
        }


# ===========================================================================
# Route Handlers
# ===========================================================================


def _get_service(request: Request) -> WOSCopilotService:
    return request.app.state.service


def _get_auth(request: Request) -> APIKeyAuth:
    return request.app.state.auth


def _get_rate_limiter(request: Request) -> RateLimiter:
    return request.app.state.rate_limiter


def _check_auth(request: Request) -> JSONResponse | None:
    auth = _get_auth(request)
    if not auth.is_valid(request):
        return JSONResponse(
            {"error": "Unauthorized", "detail": "Invalid or missing API key"},
            status_code=401,
        )
    return None


def _check_rate_limit(request: Request, org_id: str) -> JSONResponse | None:
    limiter = _get_rate_limiter(request)
    if not limiter.is_allowed(org_id):
        return JSONResponse(
            {"error": "Too Many Requests", "detail": "Rate limit exceeded"},
            status_code=429,
        )
    return None


async def handle_health(request: Request) -> JSONResponse:
    service = _get_service(request)
    return JSONResponse(service.health())


async def handle_events(request: Request) -> JSONResponse:
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"status": "accepted"}, status_code=200)

    org_id = body.get("organization_id", "unknown")
    rate_err = _check_rate_limit(request, org_id)
    if rate_err:
        return rate_err

    service = _get_service(request)
    result = service.handle_event(body)
    return JSONResponse(result, status_code=200)


async def handle_recommend(request: Request) -> JSONResponse:
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"error": "Bad Request", "detail": "Invalid JSON body"},
            status_code=400,
        )

    skill = body.get("skill")
    context = body.get("context")
    request_id = body.get("request_id")

    if not skill or context is None:
        return JSONResponse(
            {"error": "Bad Request", "detail": "Missing 'skill' or 'context' fields"},
            status_code=400,
        )

    org_id = context.get("organization_id", body.get("organization_id", "unknown"))
    rate_err = _check_rate_limit(request, org_id)
    if rate_err:
        return rate_err

    service = _get_service(request)
    try:
        result = service.handle_recommend(skill, context, request_id)
        return JSONResponse(result, status_code=200)
    except KeyError as exc:
        return JSONResponse(
            {"error": "Not Found", "detail": str(exc)},
            status_code=404,
        )
    except ValueError as exc:
        return JSONResponse(
            {"error": "Unprocessable Entity", "detail": str(exc)},
            status_code=422,
        )


async def handle_feedback(request: Request) -> JSONResponse:
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"error": "Bad Request", "detail": "Invalid JSON body"},
            status_code=400,
        )

    org_id = body.get("organization_id", "unknown")
    rate_err = _check_rate_limit(request, org_id)
    if rate_err:
        return rate_err

    service = _get_service(request)
    try:
        result = service.handle_feedback(body)
        return JSONResponse(result, status_code=200)
    except ValueError as exc:
        return JSONResponse(
            {"error": "Bad Request", "detail": str(exc)},
            status_code=400,
        )


async def handle_skill_status(request: Request) -> JSONResponse:
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    skill = request.path_params["skill"]
    service = _get_service(request)
    try:
        result = service.get_skill_status(skill)
        return JSONResponse(result, status_code=200)
    except KeyError:
        return JSONResponse(
            {"error": "Not Found", "detail": f"Unknown skill: {skill}"},
            status_code=404,
        )


async def handle_skills(request: Request) -> JSONResponse:
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    service = _get_service(request)
    return JSONResponse({"skills": service.list_skills()}, status_code=200)


# ===========================================================================
# App Factory
# ===========================================================================


def create_app(config: WOSCopilotConfig) -> Starlette:
    """Create and return a configured Starlette ASGI application."""
    routes = [
        Route("/api/v1/health", handle_health, methods=["GET"]),
        Route("/api/v1/events", handle_events, methods=["POST"]),
        Route("/api/v1/recommend", handle_recommend, methods=["POST"]),
        Route("/api/v1/feedback", handle_feedback, methods=["POST"]),
        Route("/api/v1/status/{skill}", handle_skill_status, methods=["GET"]),
        Route("/api/v1/skills", handle_skills, methods=["GET"]),
    ]

    app = Starlette(routes=routes)
    app.state.service = WOSCopilotService(config)
    app.state.auth = APIKeyAuth(config.api_key)
    app.state.rate_limiter = RateLimiter(config.rate_limit_per_minute)

    return app


def run_server(
    config: WOSCopilotConfig,
    host: str = "0.0.0.0",
    port: int = 8100,
) -> None:
    """Start the copilot HTTP server (blocking)."""
    import uvicorn

    app = create_app(config)
    uvicorn.run(app, host=host, port=port)


__all__ = [
    "WOSCopilotConfig",
    "WOSCopilotService",
    "APIKeyAuth",
    "RateLimiter",
    "create_app",
    "run_server",
]
