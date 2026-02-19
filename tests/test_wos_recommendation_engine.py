"""Tests for apprentice.wos_recommendation_engine — skill-specific recommendations."""

import uuid
import pytest

from apprentice.wos_recommendation_engine import (
    TicketTriageRecommendation,
    GuestResponseRecommendation,
    RefundRecommendation,
    BookingModificationRecommendation,
    EscalationRecommendation,
    RecommendationEngine,
    RecommendationBuilder,
    TicketTriageBuilder,
    GuestResponseBuilder,
    RefundBuilder,
    BookingModificationBuilder,
    EscalationBuilder,
    create_recommendation_engine,
)
from apprentice.wos_skill_definitions import create_skill_registry
from apprentice.middleware import MiddlewarePipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return create_recommendation_engine()


@pytest.fixture
def triage_context():
    return {"conversation_id": "conv-1", "conversation_text": "My AC is broken"}


@pytest.fixture
def guest_response_context():
    return {
        "conversation_id": "conv-2",
        "conversation_text": "When can I check in?",
        "booking_id": "W-AB12CD34",
    }


@pytest.fixture
def refund_context():
    return {
        "conversation_id": "conv-3",
        "conversation_text": "I want a refund",
        "booking_id": "W-AB12CD34",
        "booking_value": 500.0,
    }


@pytest.fixture
def booking_mod_context():
    return {
        "conversation_id": "conv-4",
        "conversation_text": "Can I change dates?",
        "booking_id": "W-AB12CD34",
        "availability": {"2024-03-01": True},
    }


@pytest.fixture
def escalation_context():
    return {
        "conversation_id": "conv-5",
        "conversation_text": "This is outrageous, I want to speak to a manager",
    }


# ---------------------------------------------------------------------------
# Recommendation Models
# ---------------------------------------------------------------------------

class TestTicketTriageRecommendation:
    def test_create_valid(self):
        r = TicketTriageRecommendation(
            conversation_id="conv-1",
            category="complaint",
            priority="high",
            suggested_assignee="team-support",
            confidence=0.9,
            reasoning="Guest is upset",
        )
        assert r.skill_name == "ticket_triage"
        assert r.confidence == 0.9
        assert r.recommendation_id  # auto-generated

    def test_frozen(self):
        r = TicketTriageRecommendation(
            conversation_id="conv-1", category="x", priority="low",
            suggested_assignee="a", confidence=0.5, reasoning="r",
        )
        with pytest.raises((AttributeError, TypeError, Exception)):
            r.category = "new"


class TestGuestResponseRecommendation:
    def test_create_valid(self):
        r = GuestResponseRecommendation(
            conversation_id="conv-1",
            suggested_text="Hello!",
            tone="professional",
            confidence=0.8,
        )
        assert r.skill_name == "guest_response"
        assert r.referenced_booking_details == {}


class TestRefundRecommendation:
    def test_create_valid(self):
        r = RefundRecommendation(
            conversation_id="conv-1",
            action="approve",
            amount=100.0,
            category="service_issue",
            reasoning="Valid claim",
            policy_reference="standard",
            confidence=0.7,
        )
        assert r.skill_name == "refund_handling"
        assert r.amount == 100.0


class TestBookingModificationRecommendation:
    def test_create_valid(self):
        r = BookingModificationRecommendation(
            conversation_id="conv-1",
            mod_type="date_change",
            changes={"check_in": "2024-03-15"},
            impact_summary="No fee",
            confidence=0.6,
        )
        assert r.skill_name == "booking_modification"


class TestEscalationRecommendation:
    def test_create_valid(self):
        r = EscalationRecommendation(
            conversation_id="conv-1",
            should_escalate=True,
            urgency="high",
            reason="Guest is angry",
            suggested_team="management",
            confidence=0.85,
        )
        assert r.skill_name == "escalation_detection"
        assert r.should_escalate is True


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

class TestBuilders:
    def test_triage_builder(self):
        b = TicketTriageBuilder()
        result = b.build({"conversation_id": "c1", "conversation_text": "hi"})
        assert "category" in result
        assert "priority" in result
        assert "confidence" in result
        assert result["conversation_id"] == "c1"

    def test_guest_response_builder(self):
        b = GuestResponseBuilder()
        result = b.build({"conversation_id": "c1"})
        assert "suggested_text" in result
        assert "tone" in result

    def test_refund_builder(self):
        b = RefundBuilder()
        result = b.build({"conversation_id": "c1"})
        assert "action" in result
        assert "amount" in result

    def test_booking_mod_builder(self):
        b = BookingModificationBuilder()
        result = b.build({"conversation_id": "c1"})
        assert "mod_type" in result
        assert "changes" in result

    def test_escalation_builder(self):
        b = EscalationBuilder()
        result = b.build({"conversation_id": "c1"})
        assert "should_escalate" in result
        assert "urgency" in result

    def test_builder_protocol(self):
        for builder_cls in [TicketTriageBuilder, GuestResponseBuilder, RefundBuilder,
                            BookingModificationBuilder, EscalationBuilder]:
            b = builder_cls()
            assert isinstance(b, RecommendationBuilder)


# ---------------------------------------------------------------------------
# Engine — happy path
# ---------------------------------------------------------------------------

class TestEngineHappyPath:
    def test_triage_recommend(self, engine, triage_context):
        rec = engine.recommend("ticket_triage", triage_context)
        assert isinstance(rec, TicketTriageRecommendation)
        assert rec.skill_name == "ticket_triage"
        assert 0.0 <= rec.confidence <= 1.0

    def test_guest_response_recommend(self, engine, guest_response_context):
        rec = engine.recommend("guest_response", guest_response_context)
        assert isinstance(rec, GuestResponseRecommendation)

    def test_refund_recommend(self, engine, refund_context):
        rec = engine.recommend("refund_handling", refund_context)
        assert isinstance(rec, RefundRecommendation)

    def test_booking_mod_recommend(self, engine, booking_mod_context):
        rec = engine.recommend("booking_modification", booking_mod_context)
        assert isinstance(rec, BookingModificationRecommendation)

    def test_escalation_recommend(self, engine, escalation_context):
        rec = engine.recommend("escalation_detection", escalation_context)
        assert isinstance(rec, EscalationRecommendation)

    def test_list_skills(self, engine):
        skills = engine.list_skills()
        assert len(skills) == 5
        assert "ticket_triage" in skills

    def test_custom_request_id(self, engine, triage_context):
        rec = engine.recommend("ticket_triage", triage_context, request_id="custom-123")
        assert rec.recommendation_id  # still has one


# ---------------------------------------------------------------------------
# Engine — error cases
# ---------------------------------------------------------------------------

class TestEngineErrors:
    def test_unknown_skill(self, engine):
        with pytest.raises(KeyError):
            engine.recommend("nonexistent", {})

    def test_missing_context_fields(self, engine):
        with pytest.raises(ValueError, match="Missing required context"):
            engine.recommend("ticket_triage", {"conversation_id": "c1"})

    def test_empty_context(self, engine):
        with pytest.raises(ValueError):
            engine.recommend("ticket_triage", {})

    def test_refund_missing_booking_value(self, engine):
        with pytest.raises(ValueError):
            engine.recommend("refund_handling", {
                "conversation_id": "c1",
                "conversation_text": "refund",
                "booking_id": "b1",
            })


# ---------------------------------------------------------------------------
# Engine — middleware integration
# ---------------------------------------------------------------------------

class TestEngineMiddleware:
    def test_with_empty_pipeline(self, triage_context):
        engine = RecommendationEngine(
            skill_registry=create_skill_registry(),
            middleware_pipeline=MiddlewarePipeline(),
        )
        rec = engine.recommend("ticket_triage", triage_context)
        assert isinstance(rec, TicketTriageRecommendation)

    def test_with_none_pipeline(self, triage_context):
        engine = RecommendationEngine(
            skill_registry=create_skill_registry(),
            middleware_pipeline=None,
        )
        rec = engine.recommend("ticket_triage", triage_context)
        assert isinstance(rec, TicketTriageRecommendation)


# ---------------------------------------------------------------------------
# Engine — custom builders
# ---------------------------------------------------------------------------

class TestEngineCustomBuilders:
    def test_custom_builder(self, triage_context):
        class MyBuilder:
            def build(self, context: dict) -> dict:
                return {
                    "conversation_id": context["conversation_id"],
                    "category": "custom_category",
                    "priority": "urgent",
                    "suggested_assignee": "custom_agent",
                    "confidence": 0.99,
                    "reasoning": "Custom builder",
                }

        engine = RecommendationEngine(
            skill_registry=create_skill_registry(),
            builders={"ticket_triage": MyBuilder()},
        )
        rec = engine.recommend("ticket_triage", triage_context)
        assert rec.category == "custom_category"
        assert rec.confidence == 0.99

    def test_missing_builder_raises(self, triage_context):
        engine = RecommendationEngine(
            skill_registry=create_skill_registry(),
            builders={},
        )
        with pytest.raises(KeyError, match="No builder"):
            engine.recommend("ticket_triage", triage_context)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_create_default(self):
        engine = create_recommendation_engine()
        assert len(engine.list_skills()) == 5

    def test_create_with_pipeline(self):
        engine = create_recommendation_engine(
            middleware_pipeline=MiddlewarePipeline()
        )
        assert len(engine.list_skills()) == 5

    def test_all_skills_produce_recommendations(self):
        engine = create_recommendation_engine()
        contexts = {
            "ticket_triage": {"conversation_id": "c", "conversation_text": "t"},
            "guest_response": {"conversation_id": "c", "conversation_text": "t", "booking_id": "b"},
            "refund_handling": {"conversation_id": "c", "conversation_text": "t", "booking_id": "b", "booking_value": 100},
            "booking_modification": {"conversation_id": "c", "conversation_text": "t", "booking_id": "b", "availability": {}},
            "escalation_detection": {"conversation_id": "c", "conversation_text": "t"},
        }
        for skill, ctx in contexts.items():
            rec = engine.recommend(skill, ctx)
            assert rec.skill_name == skill
