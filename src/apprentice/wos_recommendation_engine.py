"""
WOS Recommendation Engine — generates structured, skill-specific recommendations.

Each of the 5 WOS skills has a typed recommendation model and a builder.
The engine routes requests to the appropriate builder, applies middleware,
and returns typed recommendations.
"""

import uuid
from typing import Protocol, Union, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from apprentice.middleware import (
    MiddlewareContext,
    MiddlewarePipeline,
    MiddlewareResponse,
)
from apprentice.wos_skill_definitions import WOSSkillRegistry, create_skill_registry


# ===========================================================================
# Recommendation Models
# ===========================================================================


class TicketTriageRecommendation(BaseModel):
    model_config = ConfigDict(frozen=True)

    recommendation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    category: str
    priority: str
    suggested_assignee: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    skill_name: str = "ticket_triage"


class GuestResponseRecommendation(BaseModel):
    model_config = ConfigDict(frozen=True)

    recommendation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    suggested_text: str
    tone: str
    referenced_booking_details: dict = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    skill_name: str = "guest_response"


class RefundRecommendation(BaseModel):
    model_config = ConfigDict(frozen=True)

    recommendation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    action: str
    amount: float = Field(ge=0.0)
    category: str
    reasoning: str
    policy_reference: str
    confidence: float = Field(ge=0.0, le=1.0)
    skill_name: str = "refund_handling"


class BookingModificationRecommendation(BaseModel):
    model_config = ConfigDict(frozen=True)

    recommendation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    mod_type: str
    changes: dict = Field(default_factory=dict)
    impact_summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    skill_name: str = "booking_modification"


class EscalationRecommendation(BaseModel):
    model_config = ConfigDict(frozen=True)

    recommendation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    should_escalate: bool
    urgency: str
    reason: str
    suggested_team: str
    confidence: float = Field(ge=0.0, le=1.0)
    skill_name: str = "escalation_detection"


Recommendation = Union[
    TicketTriageRecommendation,
    GuestResponseRecommendation,
    RefundRecommendation,
    BookingModificationRecommendation,
    EscalationRecommendation,
]

_RECOMMENDATION_MODELS: dict[str, type[BaseModel]] = {
    "ticket_triage": TicketTriageRecommendation,
    "guest_response": GuestResponseRecommendation,
    "refund_handling": RefundRecommendation,
    "booking_modification": BookingModificationRecommendation,
    "escalation_detection": EscalationRecommendation,
}


# ===========================================================================
# Builder Protocol & Placeholder Builders
# ===========================================================================


@runtime_checkable
class RecommendationBuilder(Protocol):
    def build(self, context: dict) -> dict: ...


class TicketTriageBuilder:
    def build(self, context: dict) -> dict:
        return {
            "conversation_id": context.get("conversation_id", ""),
            "category": "general_inquiry",
            "priority": "medium",
            "suggested_assignee": "unassigned",
            "confidence": 0.5,
            "reasoning": "Placeholder triage — awaiting training data.",
        }


class GuestResponseBuilder:
    def build(self, context: dict) -> dict:
        return {
            "conversation_id": context.get("conversation_id", ""),
            "suggested_text": "Thank you for reaching out. Let me look into this for you.",
            "tone": "professional",
            "referenced_booking_details": {},
            "confidence": 0.3,
        }


class RefundBuilder:
    def build(self, context: dict) -> dict:
        return {
            "conversation_id": context.get("conversation_id", ""),
            "action": "escalate",
            "amount": 0.0,
            "category": "review_required",
            "reasoning": "Placeholder — requires human review.",
            "policy_reference": "standard_refund_policy",
            "confidence": 0.2,
        }


class BookingModificationBuilder:
    def build(self, context: dict) -> dict:
        return {
            "conversation_id": context.get("conversation_id", ""),
            "mod_type": "other",
            "changes": {},
            "impact_summary": "Placeholder — requires analysis.",
            "confidence": 0.3,
        }


class EscalationBuilder:
    def build(self, context: dict) -> dict:
        return {
            "conversation_id": context.get("conversation_id", ""),
            "should_escalate": False,
            "urgency": "low",
            "reason": "No escalation indicators detected.",
            "suggested_team": "",
            "confidence": 0.6,
        }


_DEFAULT_BUILDERS: dict[str, RecommendationBuilder] = {
    "ticket_triage": TicketTriageBuilder(),
    "guest_response": GuestResponseBuilder(),
    "refund_handling": RefundBuilder(),
    "booking_modification": BookingModificationBuilder(),
    "escalation_detection": EscalationBuilder(),
}


# ===========================================================================
# Recommendation Engine
# ===========================================================================


class RecommendationEngine:
    """Routes skill requests to builders, applies middleware, returns typed recommendations."""

    def __init__(
        self,
        skill_registry: WOSSkillRegistry,
        middleware_pipeline: MiddlewarePipeline | None = None,
        builders: dict[str, RecommendationBuilder] | None = None,
    ) -> None:
        self._registry = skill_registry
        self._pipeline = middleware_pipeline or MiddlewarePipeline()
        self._builders = builders if builders is not None else dict(_DEFAULT_BUILDERS)

    def recommend(
        self,
        skill_name: str,
        context: dict,
        request_id: str | None = None,
    ) -> Recommendation:
        """Generate a recommendation for the given skill.

        Raises KeyError if skill unknown, ValueError if context missing required fields.
        """
        skill_config = self._registry.get_skill(skill_name)

        is_valid, missing = self._registry.validate_context(skill_name, context)
        if not is_valid:
            raise ValueError(
                f"Missing required context fields for '{skill_name}': {missing}"
            )

        if request_id is None:
            request_id = str(uuid.uuid4())

        # Pre-process through middleware
        mw_ctx = MiddlewareContext(
            request_id=request_id,
            task_name=skill_name,
            input_data=context,
        )
        processed_ctx = self._pipeline.execute_pre(mw_ctx)

        # Build recommendation
        builder = self._builders.get(skill_name)
        if builder is None:
            raise KeyError(f"No builder registered for skill '{skill_name}'")

        raw_output = builder.build(processed_ctx.input_data)

        # Post-process through middleware
        mw_resp = MiddlewareResponse(output_data=raw_output)
        processed_resp = self._pipeline.execute_post(processed_ctx, mw_resp)

        # Create typed recommendation
        model_cls = _RECOMMENDATION_MODELS[skill_name]
        return model_cls(**processed_resp.output_data)

    def list_skills(self) -> list[str]:
        return self._registry.list_skills()


# ===========================================================================
# Factory
# ===========================================================================


def create_recommendation_engine(
    middleware_pipeline: MiddlewarePipeline | None = None,
) -> RecommendationEngine:
    """Create engine with default skill registry and builders."""
    return RecommendationEngine(
        skill_registry=create_skill_registry(),
        middleware_pipeline=middleware_pipeline,
    )


__all__ = [
    "TicketTriageRecommendation",
    "GuestResponseRecommendation",
    "RefundRecommendation",
    "BookingModificationRecommendation",
    "EscalationRecommendation",
    "Recommendation",
    "RecommendationBuilder",
    "RecommendationEngine",
    "create_recommendation_engine",
]
