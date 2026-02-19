"""
Contract tests for the WOS Event Adapter component.

Sections:
1. Fixtures and helpers
2. WOSEventType enum tests
3. WOSEvent model tests (happy + validation errors)
4. WOSEventAdapter skill mapping tests (parametrized)
5. WOSEventAdapter EventType mapping tests (parametrized)
6. WOSEventAdapter.adapt integration tests
7. WOSContextBuilder tests (add, get, rolling window, org boundary, summary, clear)
8. validate_wos_event tests (happy + error paths)
"""

import uuid

import pytest
from pydantic import ValidationError

from apprentice.observer import EventType, ObservationEvent
from apprentice.wos_event_adapter import (
    WOSContextBuilder,
    WOSEvent,
    WOSEventAdapter,
    WOSEventType,
    validate_wos_event,
)


# ===========================================================================
# FIXTURES & HELPERS
# ===========================================================================


def make_wos_event(
    event_type: WOSEventType = WOSEventType.message_received,
    conversation_id: str = "conv-001",
    agent_id: str = "agent-1",
    organization_id: str = "org-100",
    timestamp: str = "2026-02-18T12:00:00Z",
    payload: dict | None = None,
    booking_id: str = "",
    guest_id: str = "",
) -> WOSEvent:
    return WOSEvent(
        event_type=event_type,
        conversation_id=conversation_id,
        agent_id=agent_id,
        organization_id=organization_id,
        timestamp=timestamp,
        payload=payload or {},
        booking_id=booking_id,
        guest_id=guest_id,
    )


@pytest.fixture
def adapter():
    return WOSEventAdapter()


@pytest.fixture
def context_builder():
    return WOSContextBuilder(max_events_per_conversation=100)


@pytest.fixture
def small_context_builder():
    return WOSContextBuilder(max_events_per_conversation=3)


# ===========================================================================
# WOSEventType ENUM
# ===========================================================================


class TestWOSEventType:
    def test_member_count(self):
        assert len(WOSEventType) == 9

    def test_all_values(self):
        expected = {
            "message_sent",
            "message_received",
            "conversation_status_changed",
            "task_created",
            "task_updated",
            "refund_requested",
            "refund_decided",
            "booking_modified",
            "agent_assigned",
        }
        assert {e.value for e in WOSEventType} == expected

    def test_is_str_enum(self):
        for member in WOSEventType:
            assert isinstance(member, str)

    def test_individual_values(self):
        assert WOSEventType.message_sent == "message_sent"
        assert WOSEventType.message_received == "message_received"
        assert WOSEventType.conversation_status_changed == "conversation_status_changed"
        assert WOSEventType.task_created == "task_created"
        assert WOSEventType.task_updated == "task_updated"
        assert WOSEventType.refund_requested == "refund_requested"
        assert WOSEventType.refund_decided == "refund_decided"
        assert WOSEventType.booking_modified == "booking_modified"
        assert WOSEventType.agent_assigned == "agent_assigned"


# ===========================================================================
# WOSEvent MODEL — HAPPY PATH
# ===========================================================================


class TestWOSEventModel:
    def test_creation_with_required_fields(self):
        event = WOSEvent(
            event_type=WOSEventType.message_sent,
            conversation_id="conv-1",
            organization_id="org-1",
            timestamp="2026-01-01T00:00:00Z",
        )
        assert event.event_type == WOSEventType.message_sent
        assert event.conversation_id == "conv-1"
        assert event.organization_id == "org-1"
        assert event.timestamp == "2026-01-01T00:00:00Z"
        assert event.agent_id == ""
        assert event.payload == {}
        assert event.booking_id == ""
        assert event.guest_id == ""

    def test_creation_with_all_fields(self):
        event = make_wos_event(
            event_type=WOSEventType.refund_requested,
            conversation_id="conv-99",
            agent_id="agent-42",
            organization_id="org-7",
            timestamp="2026-02-18T15:30:00Z",
            payload={"amount": 150.00, "reason": "damaged"},
            booking_id="bk-500",
            guest_id="guest-88",
        )
        assert event.event_type == WOSEventType.refund_requested
        assert event.agent_id == "agent-42"
        assert event.payload == {"amount": 150.00, "reason": "damaged"}
        assert event.booking_id == "bk-500"
        assert event.guest_id == "guest-88"

    def test_frozen(self):
        event = make_wos_event()
        with pytest.raises(Exception):
            event.conversation_id = "changed"

    def test_payload_default_factory_isolation(self):
        e1 = make_wos_event()
        e2 = make_wos_event()
        assert e1.payload is not e2.payload


# ===========================================================================
# WOSEvent MODEL — VALIDATION ERRORS
# ===========================================================================


class TestWOSEventValidation:
    def test_missing_conversation_id(self):
        with pytest.raises(ValidationError):
            WOSEvent(
                event_type=WOSEventType.message_sent,
                organization_id="org-1",
                timestamp="2026-01-01T00:00:00Z",
            )

    def test_empty_conversation_id(self):
        with pytest.raises(ValidationError):
            WOSEvent(
                event_type=WOSEventType.message_sent,
                conversation_id="",
                organization_id="org-1",
                timestamp="2026-01-01T00:00:00Z",
            )

    def test_missing_organization_id(self):
        with pytest.raises(ValidationError):
            WOSEvent(
                event_type=WOSEventType.message_sent,
                conversation_id="conv-1",
                timestamp="2026-01-01T00:00:00Z",
            )

    def test_empty_organization_id(self):
        with pytest.raises(ValidationError):
            WOSEvent(
                event_type=WOSEventType.message_sent,
                conversation_id="conv-1",
                organization_id="",
                timestamp="2026-01-01T00:00:00Z",
            )

    def test_missing_event_type(self):
        with pytest.raises(ValidationError):
            WOSEvent(
                conversation_id="conv-1",
                organization_id="org-1",
                timestamp="2026-01-01T00:00:00Z",
            )

    def test_invalid_event_type(self):
        with pytest.raises(ValidationError):
            WOSEvent(
                event_type="not_a_real_event",
                conversation_id="conv-1",
                organization_id="org-1",
                timestamp="2026-01-01T00:00:00Z",
            )

    def test_missing_timestamp(self):
        with pytest.raises(ValidationError):
            WOSEvent(
                event_type=WOSEventType.message_sent,
                conversation_id="conv-1",
                organization_id="org-1",
            )


# ===========================================================================
# WOSEventAdapter — SKILL MAPPING (parametrized)
# ===========================================================================


_SKILL_MAPPING_CASES = [
    (WOSEventType.message_sent, "guest_response"),
    (WOSEventType.message_received, "guest_response"),
    (WOSEventType.conversation_status_changed, "ticket_triage"),
    (WOSEventType.task_created, "ticket_triage"),
    (WOSEventType.task_updated, "ticket_triage"),
    (WOSEventType.agent_assigned, "ticket_triage"),
    (WOSEventType.refund_requested, "refund_handling"),
    (WOSEventType.refund_decided, "refund_handling"),
    (WOSEventType.booking_modified, "booking_modification"),
]


class TestAdapterSkillMapping:
    @pytest.mark.parametrize("wos_type,expected_skill", _SKILL_MAPPING_CASES)
    def test_skill_mapping(self, adapter, wos_type, expected_skill):
        event = make_wos_event(event_type=wos_type)
        result = adapter.adapt(event)
        assert result.task_name == expected_skill

    def test_all_event_types_have_skill_mapping(self, adapter):
        """Every WOSEventType produces a valid skill name."""
        for wos_type in WOSEventType:
            event = make_wos_event(event_type=wos_type)
            result = adapter.adapt(event)
            assert isinstance(result.task_name, str)
            assert len(result.task_name) > 0


# ===========================================================================
# WOSEventAdapter — EventType MAPPING (parametrized)
# ===========================================================================


_EVENT_TYPE_MAPPING_CASES = [
    (WOSEventType.message_received, EventType.user_action),
    (WOSEventType.message_sent, EventType.agent_action),
    (WOSEventType.refund_decided, EventType.agent_action),
    (WOSEventType.agent_assigned, EventType.agent_action),
    (WOSEventType.conversation_status_changed, EventType.system_event),
    (WOSEventType.task_created, EventType.system_event),
    (WOSEventType.task_updated, EventType.system_event),
    (WOSEventType.refund_requested, EventType.system_event),
    (WOSEventType.booking_modified, EventType.system_event),
]


class TestAdapterEventTypeMapping:
    @pytest.mark.parametrize("wos_type,expected_event_type", _EVENT_TYPE_MAPPING_CASES)
    def test_event_type_mapping(self, adapter, wos_type, expected_event_type):
        event = make_wos_event(event_type=wos_type)
        result = adapter.adapt(event)
        assert result.event_type == expected_event_type

    def test_all_event_types_have_event_type_mapping(self, adapter):
        """Every WOSEventType maps to a valid EventType."""
        for wos_type in WOSEventType:
            event = make_wos_event(event_type=wos_type)
            result = adapter.adapt(event)
            assert isinstance(result.event_type, EventType)


# ===========================================================================
# WOSEventAdapter — adapt() INTEGRATION
# ===========================================================================


class TestAdapterAdapt:
    def test_adapt_returns_observation_event(self, adapter):
        event = make_wos_event()
        result = adapter.adapt(event)
        assert isinstance(result, ObservationEvent)

    def test_adapt_generates_unique_event_id(self, adapter):
        event = make_wos_event()
        r1 = adapter.adapt(event)
        r2 = adapter.adapt(event)
        assert r1.event_id != r2.event_id
        uuid.UUID(r1.event_id)
        uuid.UUID(r2.event_id)

    def test_adapt_copies_payload_to_action_data(self, adapter):
        payload = {"message": "hello", "channel": "sms"}
        event = make_wos_event(payload=payload)
        result = adapter.adapt(event)
        assert result.action_data == payload

    def test_adapt_empty_payload(self, adapter):
        event = make_wos_event(payload={})
        result = adapter.adapt(event)
        assert result.action_data == {}

    def test_adapt_context_contains_conversation_id(self, adapter):
        event = make_wos_event(conversation_id="conv-xyz")
        result = adapter.adapt(event)
        assert result.context["conversation_id"] == "conv-xyz"

    def test_adapt_context_contains_organization_id(self, adapter):
        event = make_wos_event(organization_id="org-abc")
        result = adapter.adapt(event)
        assert result.context["organization_id"] == "org-abc"

    def test_adapt_context_contains_booking_id(self, adapter):
        event = make_wos_event(booking_id="bk-777")
        result = adapter.adapt(event)
        assert result.context["booking_id"] == "bk-777"

    def test_adapt_context_contains_guest_id(self, adapter):
        event = make_wos_event(guest_id="guest-55")
        result = adapter.adapt(event)
        assert result.context["guest_id"] == "guest-55"

    def test_adapt_context_has_exactly_four_keys(self, adapter):
        event = make_wos_event()
        result = adapter.adapt(event)
        assert set(result.context.keys()) == {
            "conversation_id",
            "organization_id",
            "booking_id",
            "guest_id",
        }

    def test_adapt_preserves_timestamp(self, adapter):
        ts = "2026-02-18T09:15:30Z"
        event = make_wos_event(timestamp=ts)
        result = adapter.adapt(event)
        assert result.timestamp == ts

    def test_adapt_with_nested_payload(self, adapter):
        payload = {"data": {"nested": {"deep": True}}, "list": [1, 2, 3]}
        event = make_wos_event(payload=payload)
        result = adapter.adapt(event)
        assert result.action_data == payload

    def test_adapt_message_sent_full(self, adapter):
        event = make_wos_event(
            event_type=WOSEventType.message_sent,
            conversation_id="conv-42",
            organization_id="org-10",
            timestamp="2026-02-18T10:00:00Z",
            payload={"text": "Hi there"},
            booking_id="bk-1",
            guest_id="g-1",
        )
        result = adapter.adapt(event)
        assert result.task_name == "guest_response"
        assert result.event_type == EventType.agent_action
        assert result.action_data == {"text": "Hi there"}
        assert result.context["conversation_id"] == "conv-42"
        assert result.context["organization_id"] == "org-10"
        assert result.context["booking_id"] == "bk-1"
        assert result.context["guest_id"] == "g-1"
        assert result.timestamp == "2026-02-18T10:00:00Z"

    def test_adapt_refund_requested_full(self, adapter):
        event = make_wos_event(
            event_type=WOSEventType.refund_requested,
            payload={"amount": 250.00},
        )
        result = adapter.adapt(event)
        assert result.task_name == "refund_handling"
        assert result.event_type == EventType.system_event
        assert result.action_data["amount"] == 250.00

    def test_adapt_booking_modified_full(self, adapter):
        event = make_wos_event(
            event_type=WOSEventType.booking_modified,
            booking_id="bk-999",
            payload={"check_in": "2026-03-01", "check_out": "2026-03-05"},
        )
        result = adapter.adapt(event)
        assert result.task_name == "booking_modification"
        assert result.event_type == EventType.system_event
        assert result.context["booking_id"] == "bk-999"


# ===========================================================================
# WOSContextBuilder — ADD & GET
# ===========================================================================


class TestContextBuilderAddAndGet:
    def test_add_single_event(self, context_builder):
        event = make_wos_event()
        context_builder.add_event(event)
        events = context_builder.get_context("conv-001", "org-100")
        assert len(events) == 1
        assert events[0] == event

    def test_add_multiple_events_same_conversation(self, context_builder):
        e1 = make_wos_event(timestamp="2026-01-01T00:00:00Z")
        e2 = make_wos_event(timestamp="2026-01-01T00:01:00Z")
        e3 = make_wos_event(timestamp="2026-01-01T00:02:00Z")
        context_builder.add_event(e1)
        context_builder.add_event(e2)
        context_builder.add_event(e3)
        events = context_builder.get_context("conv-001", "org-100")
        assert len(events) == 3

    def test_add_events_different_conversations(self, context_builder):
        e1 = make_wos_event(conversation_id="conv-A", organization_id="org-1")
        e2 = make_wos_event(conversation_id="conv-B", organization_id="org-2")
        context_builder.add_event(e1)
        context_builder.add_event(e2)
        assert len(context_builder.get_context("conv-A", "org-1")) == 1
        assert len(context_builder.get_context("conv-B", "org-2")) == 1

    def test_get_context_unknown_conversation(self, context_builder):
        result = context_builder.get_context("unknown", "org-1")
        assert result == []

    def test_get_context_returns_list_copy(self, context_builder):
        event = make_wos_event()
        context_builder.add_event(event)
        list1 = context_builder.get_context("conv-001", "org-100")
        list2 = context_builder.get_context("conv-001", "org-100")
        assert list1 is not list2


# ===========================================================================
# WOSContextBuilder — ROLLING WINDOW
# ===========================================================================


class TestContextBuilderRollingWindow:
    def test_enforces_max_events(self, small_context_builder):
        cb = small_context_builder
        for i in range(5):
            cb.add_event(make_wos_event(timestamp=f"2026-01-01T00:0{i}:00Z"))
        events = cb.get_context("conv-001", "org-100")
        assert len(events) == 3

    def test_oldest_events_evicted(self, small_context_builder):
        cb = small_context_builder
        timestamps = [f"2026-01-01T00:0{i}:00Z" for i in range(5)]
        for ts in timestamps:
            cb.add_event(make_wos_event(timestamp=ts))
        events = cb.get_context("conv-001", "org-100")
        assert events[0].timestamp == timestamps[2]
        assert events[1].timestamp == timestamps[3]
        assert events[2].timestamp == timestamps[4]

    def test_window_size_one(self):
        cb = WOSContextBuilder(max_events_per_conversation=1)
        cb.add_event(make_wos_event(timestamp="t1"))
        cb.add_event(make_wos_event(timestamp="t2"))
        events = cb.get_context("conv-001", "org-100")
        assert len(events) == 1
        assert events[0].timestamp == "t2"

    def test_exact_max_events(self, small_context_builder):
        cb = small_context_builder
        for i in range(3):
            cb.add_event(make_wos_event(timestamp=f"t{i}"))
        events = cb.get_context("conv-001", "org-100")
        assert len(events) == 3


# ===========================================================================
# WOSContextBuilder — ORGANIZATION BOUNDARY ENFORCEMENT
# ===========================================================================


class TestContextBuilderOrgBoundary:
    def test_add_event_org_mismatch_raises(self, context_builder):
        e1 = make_wos_event(conversation_id="conv-1", organization_id="org-A")
        e2 = make_wos_event(conversation_id="conv-1", organization_id="org-B")
        context_builder.add_event(e1)
        with pytest.raises(ValueError, match="Organization ID mismatch"):
            context_builder.add_event(e2)

    def test_get_context_org_mismatch_raises(self, context_builder):
        event = make_wos_event(conversation_id="conv-1", organization_id="org-A")
        context_builder.add_event(event)
        with pytest.raises(ValueError, match="Organization ID mismatch"):
            context_builder.get_context("conv-1", "org-WRONG")

    def test_get_context_summary_org_mismatch_raises(self, context_builder):
        event = make_wos_event(conversation_id="conv-1", organization_id="org-A")
        context_builder.add_event(event)
        with pytest.raises(ValueError, match="Organization ID mismatch"):
            context_builder.get_context_summary("conv-1", "org-WRONG")

    def test_add_event_same_org_succeeds(self, context_builder):
        e1 = make_wos_event(conversation_id="conv-1", organization_id="org-A")
        e2 = make_wos_event(conversation_id="conv-1", organization_id="org-A")
        context_builder.add_event(e1)
        context_builder.add_event(e2)
        assert len(context_builder.get_context("conv-1", "org-A")) == 2

    def test_different_conversations_different_orgs_ok(self, context_builder):
        e1 = make_wos_event(conversation_id="conv-1", organization_id="org-A")
        e2 = make_wos_event(conversation_id="conv-2", organization_id="org-B")
        context_builder.add_event(e1)
        context_builder.add_event(e2)
        assert len(context_builder.get_context("conv-1", "org-A")) == 1
        assert len(context_builder.get_context("conv-2", "org-B")) == 1


# ===========================================================================
# WOSContextBuilder — CONTEXT SUMMARY
# ===========================================================================


class TestContextBuilderSummary:
    def test_summary_unknown_conversation(self, context_builder):
        summary = context_builder.get_context_summary("unknown", "org-1")
        assert summary["event_count"] == 0
        assert summary["event_types"] == []
        assert summary["first_event_time"] is None
        assert summary["last_event_time"] is None
        assert summary["has_refund"] is False
        assert summary["has_booking_mod"] is False

    def test_summary_single_event(self, context_builder):
        event = make_wos_event(
            event_type=WOSEventType.message_received,
            timestamp="2026-02-18T10:00:00Z",
        )
        context_builder.add_event(event)
        summary = context_builder.get_context_summary("conv-001", "org-100")
        assert summary["event_count"] == 1
        assert summary["event_types"] == ["message_received"]
        assert summary["first_event_time"] == "2026-02-18T10:00:00Z"
        assert summary["last_event_time"] == "2026-02-18T10:00:00Z"
        assert summary["has_refund"] is False
        assert summary["has_booking_mod"] is False

    def test_summary_multiple_event_types(self, context_builder):
        context_builder.add_event(
            make_wos_event(event_type=WOSEventType.message_received, timestamp="t1")
        )
        context_builder.add_event(
            make_wos_event(event_type=WOSEventType.message_sent, timestamp="t2")
        )
        context_builder.add_event(
            make_wos_event(event_type=WOSEventType.task_created, timestamp="t3")
        )
        summary = context_builder.get_context_summary("conv-001", "org-100")
        assert summary["event_count"] == 3
        assert "message_received" in summary["event_types"]
        assert "message_sent" in summary["event_types"]
        assert "task_created" in summary["event_types"]

    def test_summary_preserves_event_type_order(self, context_builder):
        context_builder.add_event(
            make_wos_event(event_type=WOSEventType.task_created, timestamp="t1")
        )
        context_builder.add_event(
            make_wos_event(event_type=WOSEventType.message_received, timestamp="t2")
        )
        summary = context_builder.get_context_summary("conv-001", "org-100")
        assert summary["event_types"] == ["task_created", "message_received"]

    def test_summary_deduplicates_event_types(self, context_builder):
        for i in range(3):
            context_builder.add_event(
                make_wos_event(event_type=WOSEventType.message_received, timestamp=f"t{i}")
            )
        summary = context_builder.get_context_summary("conv-001", "org-100")
        assert summary["event_types"] == ["message_received"]

    def test_summary_first_and_last_event_time(self, context_builder):
        context_builder.add_event(make_wos_event(timestamp="2026-01-01T00:00:00Z"))
        context_builder.add_event(make_wos_event(timestamp="2026-01-01T01:00:00Z"))
        context_builder.add_event(make_wos_event(timestamp="2026-01-01T02:00:00Z"))
        summary = context_builder.get_context_summary("conv-001", "org-100")
        assert summary["first_event_time"] == "2026-01-01T00:00:00Z"
        assert summary["last_event_time"] == "2026-01-01T02:00:00Z"

    def test_summary_has_refund_requested(self, context_builder):
        context_builder.add_event(
            make_wos_event(event_type=WOSEventType.refund_requested)
        )
        summary = context_builder.get_context_summary("conv-001", "org-100")
        assert summary["has_refund"] is True

    def test_summary_has_refund_decided(self, context_builder):
        context_builder.add_event(
            make_wos_event(event_type=WOSEventType.refund_decided)
        )
        summary = context_builder.get_context_summary("conv-001", "org-100")
        assert summary["has_refund"] is True

    def test_summary_no_refund(self, context_builder):
        context_builder.add_event(
            make_wos_event(event_type=WOSEventType.message_sent)
        )
        summary = context_builder.get_context_summary("conv-001", "org-100")
        assert summary["has_refund"] is False

    def test_summary_has_booking_mod(self, context_builder):
        context_builder.add_event(
            make_wos_event(event_type=WOSEventType.booking_modified)
        )
        summary = context_builder.get_context_summary("conv-001", "org-100")
        assert summary["has_booking_mod"] is True

    def test_summary_no_booking_mod(self, context_builder):
        context_builder.add_event(
            make_wos_event(event_type=WOSEventType.task_updated)
        )
        summary = context_builder.get_context_summary("conv-001", "org-100")
        assert summary["has_booking_mod"] is False

    def test_summary_has_both_refund_and_booking(self, context_builder):
        context_builder.add_event(
            make_wos_event(event_type=WOSEventType.refund_requested, timestamp="t1")
        )
        context_builder.add_event(
            make_wos_event(event_type=WOSEventType.booking_modified, timestamp="t2")
        )
        summary = context_builder.get_context_summary("conv-001", "org-100")
        assert summary["has_refund"] is True
        assert summary["has_booking_mod"] is True

    def test_summary_keys(self, context_builder):
        context_builder.add_event(make_wos_event())
        summary = context_builder.get_context_summary("conv-001", "org-100")
        assert set(summary.keys()) == {
            "event_count",
            "event_types",
            "first_event_time",
            "last_event_time",
            "has_refund",
            "has_booking_mod",
        }


# ===========================================================================
# WOSContextBuilder — CLEAR
# ===========================================================================


class TestContextBuilderClear:
    def test_clear_removes_conversation(self, context_builder):
        context_builder.add_event(make_wos_event())
        context_builder.clear_conversation("conv-001")
        assert context_builder.get_context("conv-001", "org-100") == []

    def test_clear_nonexistent_is_safe(self, context_builder):
        context_builder.clear_conversation("nonexistent")

    def test_clear_does_not_affect_other_conversations(self, context_builder):
        e1 = make_wos_event(conversation_id="conv-A", organization_id="org-1")
        e2 = make_wos_event(conversation_id="conv-B", organization_id="org-2")
        context_builder.add_event(e1)
        context_builder.add_event(e2)
        context_builder.clear_conversation("conv-A")
        assert context_builder.get_context("conv-A", "org-1") == []
        assert len(context_builder.get_context("conv-B", "org-2")) == 1

    def test_clear_allows_re_add_with_different_org(self, context_builder):
        e1 = make_wos_event(conversation_id="conv-1", organization_id="org-A")
        context_builder.add_event(e1)
        context_builder.clear_conversation("conv-1")
        e2 = make_wos_event(conversation_id="conv-1", organization_id="org-B")
        context_builder.add_event(e2)
        events = context_builder.get_context("conv-1", "org-B")
        assert len(events) == 1


# ===========================================================================
# validate_wos_event — HAPPY PATH
# ===========================================================================


class TestValidateWOSEventHappy:
    def test_valid_minimal_dict(self):
        data = {
            "event_type": "message_sent",
            "conversation_id": "conv-1",
            "organization_id": "org-1",
            "timestamp": "2026-01-01T00:00:00Z",
        }
        event = validate_wos_event(data)
        assert isinstance(event, WOSEvent)
        assert event.event_type == WOSEventType.message_sent
        assert event.conversation_id == "conv-1"

    def test_valid_full_dict(self):
        data = {
            "event_type": "refund_requested",
            "conversation_id": "conv-99",
            "agent_id": "agent-5",
            "organization_id": "org-10",
            "timestamp": "2026-02-18T12:00:00Z",
            "payload": {"amount": 100},
            "booking_id": "bk-1",
            "guest_id": "g-1",
        }
        event = validate_wos_event(data)
        assert event.event_type == WOSEventType.refund_requested
        assert event.payload == {"amount": 100}
        assert event.booking_id == "bk-1"

    def test_valid_all_nine_event_types(self):
        for et in WOSEventType:
            data = {
                "event_type": et.value,
                "conversation_id": "conv-1",
                "organization_id": "org-1",
                "timestamp": "2026-01-01T00:00:00Z",
            }
            event = validate_wos_event(data)
            assert event.event_type == et


# ===========================================================================
# validate_wos_event — ERROR PATHS
# ===========================================================================


class TestValidateWOSEventErrors:
    def test_empty_dict(self):
        with pytest.raises(ValidationError):
            validate_wos_event({})

    def test_missing_event_type(self):
        with pytest.raises(ValidationError):
            validate_wos_event({
                "conversation_id": "conv-1",
                "organization_id": "org-1",
                "timestamp": "2026-01-01T00:00:00Z",
            })

    def test_invalid_event_type_string(self):
        with pytest.raises(ValidationError):
            validate_wos_event({
                "event_type": "bogus_event",
                "conversation_id": "conv-1",
                "organization_id": "org-1",
                "timestamp": "2026-01-01T00:00:00Z",
            })

    def test_missing_conversation_id(self):
        with pytest.raises(ValidationError):
            validate_wos_event({
                "event_type": "message_sent",
                "organization_id": "org-1",
                "timestamp": "2026-01-01T00:00:00Z",
            })

    def test_empty_conversation_id(self):
        with pytest.raises(ValidationError):
            validate_wos_event({
                "event_type": "message_sent",
                "conversation_id": "",
                "organization_id": "org-1",
                "timestamp": "2026-01-01T00:00:00Z",
            })

    def test_missing_organization_id(self):
        with pytest.raises(ValidationError):
            validate_wos_event({
                "event_type": "message_sent",
                "conversation_id": "conv-1",
                "timestamp": "2026-01-01T00:00:00Z",
            })

    def test_empty_organization_id(self):
        with pytest.raises(ValidationError):
            validate_wos_event({
                "event_type": "message_sent",
                "conversation_id": "conv-1",
                "organization_id": "",
                "timestamp": "2026-01-01T00:00:00Z",
            })

    def test_missing_timestamp(self):
        with pytest.raises(ValidationError):
            validate_wos_event({
                "event_type": "message_sent",
                "conversation_id": "conv-1",
                "organization_id": "org-1",
            })

    def test_non_dict_input(self):
        with pytest.raises((ValidationError, TypeError, AttributeError)):
            validate_wos_event("not a dict")

    def test_payload_not_a_dict(self):
        with pytest.raises(ValidationError):
            validate_wos_event({
                "event_type": "message_sent",
                "conversation_id": "conv-1",
                "organization_id": "org-1",
                "timestamp": "2026-01-01T00:00:00Z",
                "payload": "not_a_dict",
            })
