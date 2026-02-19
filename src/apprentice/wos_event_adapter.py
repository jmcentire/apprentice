"""
WOS Event Adapter (wos_event_adapter) v1

Receives raw HTTP event payloads from WOS (Wander Operating System) and
normalizes them into Apprentice ObservationEvent instances.

Components:
- WOSEventType: Enum of the 9 event types that WOS can emit.
- WOSEvent: Frozen Pydantic model representing the raw HTTP payload.
- WOSEventAdapter: Stateless transformer from WOSEvent -> ObservationEvent.
- WOSContextBuilder: Per-conversation rolling window of WOS events.
- validate_wos_event: Parse a raw dict into a validated WOSEvent.
"""

from collections import deque
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from apprentice.observer import EventType, ObservationEvent


# ===========================================================================
# Enums
# ===========================================================================


class WOSEventType(str, Enum):
    """Event types emitted by the Wander Operating System."""

    message_sent = "message_sent"
    message_received = "message_received"
    conversation_status_changed = "conversation_status_changed"
    task_created = "task_created"
    task_updated = "task_updated"
    refund_requested = "refund_requested"
    refund_decided = "refund_decided"
    booking_modified = "booking_modified"
    agent_assigned = "agent_assigned"


# ===========================================================================
# Data Models
# ===========================================================================


class WOSEvent(BaseModel):
    """Frozen Pydantic model representing a raw WOS HTTP event payload."""

    model_config = ConfigDict(frozen=True)

    event_type: WOSEventType
    conversation_id: str = Field(min_length=1)
    agent_id: str = ""
    organization_id: str = Field(min_length=1)
    timestamp: str
    payload: dict = Field(default_factory=dict)
    booking_id: str = ""
    guest_id: str = ""


# ===========================================================================
# Skill & EventType Mappings
# ===========================================================================

_SKILL_MAP: dict[WOSEventType, str] = {
    WOSEventType.message_sent: "guest_response",
    WOSEventType.message_received: "guest_response",
    WOSEventType.conversation_status_changed: "ticket_triage",
    WOSEventType.task_created: "ticket_triage",
    WOSEventType.task_updated: "ticket_triage",
    WOSEventType.agent_assigned: "ticket_triage",
    WOSEventType.refund_requested: "refund_handling",
    WOSEventType.refund_decided: "refund_handling",
    WOSEventType.booking_modified: "booking_modification",
}

_EVENT_TYPE_MAP: dict[WOSEventType, EventType] = {
    WOSEventType.message_received: EventType.user_action,
    WOSEventType.message_sent: EventType.agent_action,
    WOSEventType.refund_decided: EventType.agent_action,
    WOSEventType.agent_assigned: EventType.agent_action,
    # Everything else maps to system_event
    WOSEventType.conversation_status_changed: EventType.system_event,
    WOSEventType.task_created: EventType.system_event,
    WOSEventType.task_updated: EventType.system_event,
    WOSEventType.refund_requested: EventType.system_event,
    WOSEventType.booking_modified: EventType.system_event,
}


# ===========================================================================
# WOSEventAdapter
# ===========================================================================


class WOSEventAdapter:
    """Stateless transformer that converts a WOSEvent into an ObservationEvent.

    Maps WOS event types to Apprentice skill names and observation event types,
    copies the payload into action_data, and builds a context dict from the
    conversation metadata.
    """

    def adapt(self, event: WOSEvent) -> ObservationEvent:
        """Transform a single WOSEvent into an ObservationEvent."""
        task_name = _SKILL_MAP[event.event_type]
        event_type = _EVENT_TYPE_MAP[event.event_type]

        context = {
            "conversation_id": event.conversation_id,
            "organization_id": event.organization_id,
            "booking_id": event.booking_id,
            "guest_id": event.guest_id,
        }

        return ObservationEvent(
            task_name=task_name,
            event_type=event_type,
            action_data=event.payload,
            context=context,
            timestamp=event.timestamp,
        )


# ===========================================================================
# WOSContextBuilder
# ===========================================================================


class WOSContextBuilder:
    """Per-conversation rolling window of WOS events.

    Tracks events keyed by conversation_id and enforces that all events
    for a given conversation belong to the same organization_id.
    """

    def __init__(self, max_events_per_conversation: int = 100) -> None:
        self._max_events = max_events_per_conversation
        # conversation_id -> (organization_id, deque[WOSEvent])
        self._conversations: dict[str, tuple[str, deque[WOSEvent]]] = {}

    def add_event(self, event: WOSEvent) -> None:
        """Add a WOSEvent to the rolling window for its conversation.

        Raises ValueError if the event's organization_id does not match the
        organization_id already recorded for that conversation.
        """
        cid = event.conversation_id

        if cid in self._conversations:
            stored_org, events = self._conversations[cid]
            if event.organization_id != stored_org:
                raise ValueError(
                    f"Organization ID mismatch for conversation '{cid}': "
                    f"expected '{stored_org}', got '{event.organization_id}'"
                )
            events.append(event)
        else:
            events = deque(maxlen=self._max_events)
            events.append(event)
            self._conversations[cid] = (event.organization_id, events)

    def get_context(self, conversation_id: str, organization_id: str) -> list[WOSEvent]:
        """Return the event window for a conversation.

        Raises ValueError if the stored organization_id does not match the
        provided one.  Returns an empty list if the conversation is unknown.
        """
        if conversation_id not in self._conversations:
            return []

        stored_org, events = self._conversations[conversation_id]
        if organization_id != stored_org:
            raise ValueError(
                f"Organization ID mismatch for conversation '{conversation_id}': "
                f"expected '{stored_org}', got '{organization_id}'"
            )
        return list(events)

    def get_context_summary(self, conversation_id: str, organization_id: str) -> dict:
        """Return a summary dict for the conversation's event window.

        Keys: event_count, event_types, first_event_time, last_event_time,
              has_refund, has_booking_mod.

        Raises ValueError on org mismatch.  Returns a zeroed summary if the
        conversation is unknown.
        """
        events = self.get_context(conversation_id, organization_id)

        if not events:
            return {
                "event_count": 0,
                "event_types": [],
                "first_event_time": None,
                "last_event_time": None,
                "has_refund": False,
                "has_booking_mod": False,
            }

        event_types = list(dict.fromkeys(e.event_type.value for e in events))

        refund_types = {WOSEventType.refund_requested, WOSEventType.refund_decided}
        has_refund = any(e.event_type in refund_types for e in events)
        has_booking_mod = any(e.event_type == WOSEventType.booking_modified for e in events)

        return {
            "event_count": len(events),
            "event_types": event_types,
            "first_event_time": events[0].timestamp,
            "last_event_time": events[-1].timestamp,
            "has_refund": has_refund,
            "has_booking_mod": has_booking_mod,
        }

    def clear_conversation(self, conversation_id: str) -> None:
        """Remove all events for a conversation."""
        self._conversations.pop(conversation_id, None)


# ===========================================================================
# Validation Helper
# ===========================================================================


def validate_wos_event(data: dict) -> WOSEvent:
    """Parse and validate a raw dict into a WOSEvent.

    Raises pydantic.ValidationError on invalid input.
    """
    return WOSEvent.model_validate(data)


# ===========================================================================
# Exports
# ===========================================================================

__all__ = [
    "WOSEventType",
    "WOSEvent",
    "WOSEventAdapter",
    "WOSContextBuilder",
    "validate_wos_event",
]
