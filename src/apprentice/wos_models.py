"""WOS Integration Data Models.

Pydantic models matching what Wander Operating System (WOS) sends to Apprentice.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class WOSEventType(str, Enum):
    """Event types emitted by WOS."""
    message_sent = "message_sent"
    message_received = "message_received"
    conversation_status_changed = "conversation_status_changed"
    task_created = "task_created"
    task_updated = "task_updated"
    refund_requested = "refund_requested"
    refund_decided = "refund_decided"
    booking_modified = "booking_modified"
    agent_assigned = "agent_assigned"


class FeedbackType(str, Enum):
    """Feedback types from agents."""
    accept = "accept"
    reject = "reject"
    edit = "edit"
    ignore = "ignore"
    ai_score = "ai_score"


# Skill names are free-form strings — each consumer's apprentice.yaml
# defines what skill names mean. Apprentice itself imposes no closed
# vocabulary; the value just has to match a registered task in the
# consumer's configuration. Pre-defined Wander-OS skill name constants
# are exported below as a convenience for WOS callers and the WOS test
# fixtures, but new integrations (Reeve, third-party agents) supply
# their own strings.
SkillName = str

# Convenience constants for the original Wander-OS integration. New
# integrations should not rely on this list — register skills in your
# own apprentice.yaml instead.
WOS_SKILL_TICKET_TRIAGE: SkillName = "ticket_triage"
WOS_SKILL_GUEST_RESPONSE: SkillName = "guest_response"
WOS_SKILL_REFUND_HANDLING: SkillName = "refund_handling"
WOS_SKILL_BOOKING_MODIFICATION: SkillName = "booking_modification"
WOS_SKILL_ESCALATION_DETECTION: SkillName = "escalation_detection"


# ---------------------------------------------------------------------------
# Event type -> task_id mapping
# ---------------------------------------------------------------------------

EVENT_TYPE_TO_TASK: dict[str, str] = {
    "message_sent": "guest_response",
    "message_received": "guest_response",
    "conversation_status_changed": "ticket_triage",
    "task_created": "ticket_triage",
    "task_updated": "ticket_triage",
    "refund_requested": "refund_handling",
    "refund_decided": "refund_handling",
    "booking_modified": "booking_modification",
    "agent_assigned": "escalation_detection",
}

# Feedback type -> numeric score
FEEDBACK_SCORES: dict[str, float] = {
    "accept": 1.0,
    "reject": 0.0,
    "edit": 0.5,
    "ignore": 0.25,
    "ai_score": 0.75,
}


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class WOSEvent(BaseModel):
    """Event payload sent from WOS to Apprentice."""
    model_config = ConfigDict(extra="allow")

    event_type: WOSEventType
    conversation_id: str = Field(min_length=1)
    agent_id: str = ""
    organization_id: str = ""
    timestamp: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    booking_id: str = ""
    guest_id: str = ""


class FeedbackRecord(BaseModel):
    """Feedback on a recommendation from an agent."""
    model_config = ConfigDict(extra="allow")

    request_id: str = Field(min_length=1)
    skill: SkillName
    feedback_type: FeedbackType
    edited_output: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None


class RecommendRequest(BaseModel):
    """Request for a recommendation from Apprentice."""
    model_config = ConfigDict(extra="allow")

    skill: SkillName
    context: Dict[str, Any] = Field(default_factory=dict)
    request_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class RecommendResult(BaseModel):
    """Recommendation returned by Apprentice."""
    model_config = ConfigDict(extra="allow")

    skill_name: str
    recommendation_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    output: Dict[str, Any] = Field(default_factory=dict)


class EventResponse(BaseModel):
    """Response to an event ingestion request."""
    status: str = "accepted"


class FeedbackResponse(BaseModel):
    """Response to a feedback submission."""
    status: str = "recorded"
    match_score: Optional[float] = None


class SkillInfo(BaseModel):
    """Information about a configured skill."""
    name: str
    phase: str
    confidence: Optional[float] = None
