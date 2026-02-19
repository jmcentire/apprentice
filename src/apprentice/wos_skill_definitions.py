"""WOS Skill Definitions — defines 5 WOS-specific skills as Apprentice task configurations.

Each skill has required context fields, output fields, autonomy thresholds, risk levels,
and associated evaluators (TriageEvaluator, RefundEvaluator) for comparing local vs remote
task responses.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from apprentice.evaluators import (
    EvaluatorProtocol,
    EvaluationResult,
    FieldEvaluation,
    TaskResponse,
    TaskEvaluatorConfig,
    clamp_score,
)


# ============================================================================
# Enums
# ============================================================================


class RiskLevel(str, Enum):
    """Risk classification for WOS skills."""
    low = "low"
    medium = "medium"
    high = "high"


# ============================================================================
# Config Model
# ============================================================================


class WOSSkillConfig(BaseModel):
    """Frozen configuration for a single WOS skill."""
    model_config = ConfigDict(frozen=True)

    skill_name: str
    description: str
    required_context_fields: list[str]
    output_fields: list[str]
    autonomy_threshold: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    os_task_categories: list[str]
    evaluator_type: str = "structured_match"
    match_fields: list[str]


# ============================================================================
# Registry
# ============================================================================


class WOSSkillRegistry:
    """Registry for looking up, filtering, and validating WOS skill configurations."""

    def __init__(self, skills: list[WOSSkillConfig] | None = None) -> None:
        self._skills: dict[str, WOSSkillConfig] = {}
        if skills is not None:
            for skill in skills:
                self._skills[skill.skill_name] = skill

    def get_skill(self, skill_name: str) -> WOSSkillConfig:
        """Return the config for *skill_name*, or raise KeyError."""
        if skill_name not in self._skills:
            raise KeyError(f"Unknown skill: {skill_name}")
        return self._skills[skill_name]

    def list_skills(self) -> list[str]:
        """Return sorted list of registered skill names."""
        return sorted(self._skills.keys())

    def get_skills_for_category(self, category: str) -> list[WOSSkillConfig]:
        """Return all skills whose os_task_categories contain *category*."""
        return [
            skill
            for skill in self._skills.values()
            if category in skill.os_task_categories
        ]

    def validate_context(
        self, skill_name: str, context: dict
    ) -> tuple[bool, list[str]]:
        """Check whether *context* supplies all required fields for *skill_name*.

        Returns (is_valid, missing_fields).
        """
        skill = self.get_skill(skill_name)
        missing = [
            field
            for field in skill.required_context_fields
            if field not in context
        ]
        return (len(missing) == 0, missing)

    def can_operate_autonomously(
        self, skill_name: str, confidence: float
    ) -> bool:
        """Return True only when risk_level is low AND confidence >= threshold."""
        skill = self.get_skill(skill_name)
        return (
            skill.risk_level == RiskLevel.low
            and confidence >= skill.autonomy_threshold
        )

    def __contains__(self, skill_name: str) -> bool:
        return skill_name in self._skills

    def __len__(self) -> int:
        return len(self._skills)


# ============================================================================
# Evaluators
# ============================================================================


class TriageEvaluator:
    """Weighted evaluator for ticket-triage tasks.

    Field weights: category (50%), priority (30%), suggested_assignee (20%).
    Each field uses exact-match scoring (1.0 or 0.0).
    """

    _WEIGHTS: dict[str, float] = {
        "category": 0.50,
        "priority": 0.30,
        "suggested_assignee": 0.20,
    }

    def evaluate(
        self,
        local: TaskResponse,
        remote: TaskResponse,
        task_config: TaskEvaluatorConfig,
    ) -> EvaluationResult:
        timestamp = datetime.now(timezone.utc).isoformat()
        field_breakdown: dict[str, FieldEvaluation] = {}
        weighted_sum = 0.0

        for field_name, weight in self._WEIGHTS.items():
            local_value = local.fields.get(field_name)
            remote_value = remote.fields.get(field_name)
            matched = local_value == remote_value
            similarity = 1.0 if matched else 0.0
            weighted_sum += weight * similarity
            field_breakdown[field_name] = FieldEvaluation(
                field_name=field_name,
                matched=matched,
                local_value=local_value,
                remote_value=remote_value,
                similarity=similarity,
            )

        score = clamp_score(weighted_sum)
        return EvaluationResult(
            score=score,
            evaluator_type="triage",
            field_breakdown=field_breakdown,
            error=None,
            metadata={},
            timestamp=timestamp,
        )


class RefundEvaluator:
    """Weighted evaluator for refund-handling tasks.

    Field weights: action (40%), amount (40%), category (20%).
    action and category use exact match; amount uses 5% tolerance.
    """

    _WEIGHTS: dict[str, float] = {
        "action": 0.40,
        "amount": 0.40,
        "category": 0.20,
    }

    @staticmethod
    def _amount_match(local_val: object, remote_val: object) -> bool:
        """Return True when the two amount values are within 5% tolerance."""
        try:
            local_f = float(local_val)  # type: ignore[arg-type]
            remote_f = float(remote_val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        denominator = max(abs(remote_f), 0.01)
        return abs(local_f - remote_f) / denominator <= 0.05

    def evaluate(
        self,
        local: TaskResponse,
        remote: TaskResponse,
        task_config: TaskEvaluatorConfig,
    ) -> EvaluationResult:
        timestamp = datetime.now(timezone.utc).isoformat()
        field_breakdown: dict[str, FieldEvaluation] = {}
        weighted_sum = 0.0

        for field_name, weight in self._WEIGHTS.items():
            local_value = local.fields.get(field_name)
            remote_value = remote.fields.get(field_name)

            if field_name == "amount":
                matched = self._amount_match(local_value, remote_value)
            else:
                matched = local_value == remote_value

            similarity = 1.0 if matched else 0.0
            weighted_sum += weight * similarity
            field_breakdown[field_name] = FieldEvaluation(
                field_name=field_name,
                matched=matched,
                local_value=local_value,
                remote_value=remote_value,
                similarity=similarity,
            )

        score = clamp_score(weighted_sum)
        return EvaluationResult(
            score=score,
            evaluator_type="refund",
            field_breakdown=field_breakdown,
            error=None,
            metadata={},
            timestamp=timestamp,
        )


# ============================================================================
# Skill Definitions
# ============================================================================


def skill_configs() -> list[WOSSkillConfig]:
    """Return the canonical list of all 5 WOS skill configurations."""
    return [
        WOSSkillConfig(
            skill_name="ticket_triage",
            description="Classify and route incoming support tickets",
            required_context_fields=["conversation_id", "conversation_text"],
            output_fields=[
                "category",
                "priority",
                "suggested_assignee",
                "confidence",
                "reasoning",
            ],
            autonomy_threshold=0.85,
            risk_level=RiskLevel.low,
            os_task_categories=[
                "general_inquiry",
                "complaint",
                "maintenance",
                "booking_issue",
                "payment_issue",
            ],
            evaluator_type="structured_match",
            match_fields=["category", "priority", "suggested_assignee"],
        ),
        WOSSkillConfig(
            skill_name="guest_response",
            description="Generate guest-facing response text",
            required_context_fields=[
                "conversation_id",
                "conversation_text",
                "booking_id",
            ],
            output_fields=[
                "suggested_text",
                "tone",
                "referenced_booking_details",
                "confidence",
            ],
            autonomy_threshold=0.95,
            risk_level=RiskLevel.high,
            os_task_categories=["guest_communication"],
            evaluator_type="semantic_similarity",
            match_fields=["tone"],
        ),
        WOSSkillConfig(
            skill_name="refund_handling",
            description="Determine refund action, amount, and categorisation",
            required_context_fields=[
                "conversation_id",
                "conversation_text",
                "booking_id",
                "booking_value",
            ],
            output_fields=[
                "action",
                "amount",
                "category",
                "reasoning",
                "policy_reference",
                "confidence",
            ],
            autonomy_threshold=0.95,
            risk_level=RiskLevel.high,
            os_task_categories=["refund", "payment_dispute"],
            evaluator_type="structured_match",
            match_fields=["action", "amount", "category"],
        ),
        WOSSkillConfig(
            skill_name="booking_modification",
            description="Evaluate and summarise proposed booking modifications",
            required_context_fields=[
                "conversation_id",
                "conversation_text",
                "booking_id",
                "availability",
            ],
            output_fields=[
                "mod_type",
                "changes",
                "impact_summary",
                "confidence",
            ],
            autonomy_threshold=0.90,
            risk_level=RiskLevel.medium,
            os_task_categories=[
                "date_change",
                "guest_count_change",
                "cancellation",
            ],
            evaluator_type="structured_match",
            match_fields=["mod_type", "changes"],
        ),
        WOSSkillConfig(
            skill_name="escalation_detection",
            description="Detect whether a conversation requires escalation",
            required_context_fields=["conversation_id", "conversation_text"],
            output_fields=[
                "should_escalate",
                "urgency",
                "reason",
                "suggested_team",
                "confidence",
            ],
            autonomy_threshold=0.80,
            risk_level=RiskLevel.low,
            os_task_categories=["escalation"],
            evaluator_type="structured_match",
            match_fields=["should_escalate", "urgency"],
        ),
    ]


# ============================================================================
# Factory
# ============================================================================


def create_skill_registry() -> WOSSkillRegistry:
    """Create a WOSSkillRegistry pre-loaded with all canonical WOS skills."""
    return WOSSkillRegistry(skills=skill_configs())
