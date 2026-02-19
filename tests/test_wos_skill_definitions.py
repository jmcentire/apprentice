"""Tests for apprentice.wos_skill_definitions — WOS skill configs, registry, and evaluators."""

import pytest
from datetime import datetime, timezone

from pydantic import ValidationError

from apprentice.wos_skill_definitions import (
    RiskLevel,
    WOSSkillConfig,
    WOSSkillRegistry,
    TriageEvaluator,
    RefundEvaluator,
    skill_configs,
    create_skill_registry,
)
from apprentice.evaluators import (
    TaskResponse,
    TaskEvaluatorConfig,
    EvaluationResult,
    FieldEvaluation,
    EvaluatorProtocol,
)


# ============================================================================
# Helpers
# ============================================================================


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_response(fields: dict, task_id: str = "t1", output: str = "") -> TaskResponse:
    return TaskResponse(
        task_id=task_id,
        output=output,
        fields=fields,
        model_id="model-1",
        timestamp=_ts(),
    )


def _triage_config() -> TaskEvaluatorConfig:
    return TaskEvaluatorConfig(
        evaluator_type="structured_match",
        match_fields=["category", "priority", "suggested_assignee"],
    )


def _refund_config() -> TaskEvaluatorConfig:
    return TaskEvaluatorConfig(
        evaluator_type="structured_match",
        match_fields=["action", "amount", "category"],
    )


# ============================================================================
# RiskLevel enum
# ============================================================================


class TestRiskLevel:
    def test_low(self):
        assert RiskLevel.low.value == "low"

    def test_medium(self):
        assert RiskLevel.medium.value == "medium"

    def test_high(self):
        assert RiskLevel.high.value == "high"

    def test_is_str_enum(self):
        assert isinstance(RiskLevel.low, str)


# ============================================================================
# WOSSkillConfig model
# ============================================================================


class TestWOSSkillConfig:
    def test_basic_construction(self):
        cfg = WOSSkillConfig(
            skill_name="test",
            description="A test skill",
            required_context_fields=["a"],
            output_fields=["b"],
            autonomy_threshold=0.5,
            risk_level=RiskLevel.low,
            os_task_categories=["cat1"],
            match_fields=["b"],
        )
        assert cfg.skill_name == "test"
        assert cfg.evaluator_type == "structured_match"

    def test_frozen(self):
        cfg = WOSSkillConfig(
            skill_name="test",
            description="desc",
            required_context_fields=[],
            output_fields=[],
            autonomy_threshold=0.5,
            risk_level=RiskLevel.low,
            os_task_categories=[],
            match_fields=[],
        )
        with pytest.raises((AttributeError, ValidationError)):
            cfg.skill_name = "changed"

    def test_threshold_lower_bound(self):
        with pytest.raises(ValidationError):
            WOSSkillConfig(
                skill_name="x",
                description="d",
                required_context_fields=[],
                output_fields=[],
                autonomy_threshold=-0.1,
                risk_level=RiskLevel.low,
                os_task_categories=[],
                match_fields=[],
            )

    def test_threshold_upper_bound(self):
        with pytest.raises(ValidationError):
            WOSSkillConfig(
                skill_name="x",
                description="d",
                required_context_fields=[],
                output_fields=[],
                autonomy_threshold=1.1,
                risk_level=RiskLevel.low,
                os_task_categories=[],
                match_fields=[],
            )

    def test_threshold_boundaries_valid(self):
        for val in (0.0, 1.0):
            cfg = WOSSkillConfig(
                skill_name="x",
                description="d",
                required_context_fields=[],
                output_fields=[],
                autonomy_threshold=val,
                risk_level=RiskLevel.low,
                os_task_categories=[],
                match_fields=[],
            )
            assert cfg.autonomy_threshold == val

    def test_invalid_risk_level(self):
        with pytest.raises((ValidationError, ValueError)):
            WOSSkillConfig(
                skill_name="x",
                description="d",
                required_context_fields=[],
                output_fields=[],
                autonomy_threshold=0.5,
                risk_level="invalid",
                os_task_categories=[],
                match_fields=[],
            )

    def test_custom_evaluator_type(self):
        cfg = WOSSkillConfig(
            skill_name="x",
            description="d",
            required_context_fields=[],
            output_fields=[],
            autonomy_threshold=0.5,
            risk_level=RiskLevel.low,
            os_task_categories=[],
            evaluator_type="semantic_similarity",
            match_fields=[],
        )
        assert cfg.evaluator_type == "semantic_similarity"


# ============================================================================
# skill_configs() — canonical list
# ============================================================================


class TestSkillConfigs:
    def test_returns_five_skills(self):
        configs = skill_configs()
        assert len(configs) == 5

    def test_skill_names(self):
        names = {c.skill_name for c in skill_configs()}
        assert names == {
            "ticket_triage",
            "guest_response",
            "refund_handling",
            "booking_modification",
            "escalation_detection",
        }

    def test_ticket_triage_config(self):
        cfg = {c.skill_name: c for c in skill_configs()}["ticket_triage"]
        assert cfg.required_context_fields == ["conversation_id", "conversation_text"]
        assert "category" in cfg.output_fields
        assert "priority" in cfg.output_fields
        assert "suggested_assignee" in cfg.output_fields
        assert "confidence" in cfg.output_fields
        assert "reasoning" in cfg.output_fields
        assert cfg.autonomy_threshold == 0.85
        assert cfg.risk_level == RiskLevel.low
        assert "general_inquiry" in cfg.os_task_categories
        assert "complaint" in cfg.os_task_categories
        assert "maintenance" in cfg.os_task_categories
        assert "booking_issue" in cfg.os_task_categories
        assert "payment_issue" in cfg.os_task_categories
        assert cfg.evaluator_type == "structured_match"
        assert cfg.match_fields == ["category", "priority", "suggested_assignee"]

    def test_guest_response_config(self):
        cfg = {c.skill_name: c for c in skill_configs()}["guest_response"]
        assert cfg.required_context_fields == [
            "conversation_id",
            "conversation_text",
            "booking_id",
        ]
        assert cfg.autonomy_threshold == 0.95
        assert cfg.risk_level == RiskLevel.high
        assert cfg.os_task_categories == ["guest_communication"]
        assert cfg.evaluator_type == "semantic_similarity"
        assert cfg.match_fields == ["tone"]

    def test_refund_handling_config(self):
        cfg = {c.skill_name: c for c in skill_configs()}["refund_handling"]
        assert "booking_value" in cfg.required_context_fields
        assert cfg.autonomy_threshold == 0.95
        assert cfg.risk_level == RiskLevel.high
        assert "refund" in cfg.os_task_categories
        assert "payment_dispute" in cfg.os_task_categories
        assert cfg.match_fields == ["action", "amount", "category"]

    def test_booking_modification_config(self):
        cfg = {c.skill_name: c for c in skill_configs()}["booking_modification"]
        assert "availability" in cfg.required_context_fields
        assert cfg.autonomy_threshold == 0.90
        assert cfg.risk_level == RiskLevel.medium
        assert "date_change" in cfg.os_task_categories
        assert "guest_count_change" in cfg.os_task_categories
        assert "cancellation" in cfg.os_task_categories
        assert cfg.match_fields == ["mod_type", "changes"]

    def test_escalation_detection_config(self):
        cfg = {c.skill_name: c for c in skill_configs()}["escalation_detection"]
        assert cfg.required_context_fields == ["conversation_id", "conversation_text"]
        assert cfg.autonomy_threshold == 0.80
        assert cfg.risk_level == RiskLevel.low
        assert cfg.os_task_categories == ["escalation"]
        assert cfg.match_fields == ["should_escalate", "urgency"]

    def test_all_configs_are_wos_skill_config(self):
        for cfg in skill_configs():
            assert isinstance(cfg, WOSSkillConfig)

    def test_all_configs_have_non_empty_description(self):
        for cfg in skill_configs():
            assert len(cfg.description) > 0

    def test_all_configs_have_non_empty_match_fields(self):
        for cfg in skill_configs():
            assert len(cfg.match_fields) > 0


# ============================================================================
# WOSSkillRegistry
# ============================================================================


class TestWOSSkillRegistry:
    @pytest.fixture
    def registry(self):
        return create_skill_registry()

    def test_len(self, registry):
        assert len(registry) == 5

    def test_contains(self, registry):
        assert "ticket_triage" in registry
        assert "unknown_skill" not in registry

    def test_get_skill(self, registry):
        skill = registry.get_skill("ticket_triage")
        assert skill.skill_name == "ticket_triage"

    def test_get_skill_unknown_raises(self, registry):
        with pytest.raises(KeyError):
            registry.get_skill("nonexistent_skill")

    def test_list_skills_sorted(self, registry):
        names = registry.list_skills()
        assert names == sorted(names)
        assert len(names) == 5

    def test_get_skills_for_category_single_match(self, registry):
        results = registry.get_skills_for_category("escalation")
        assert len(results) == 1
        assert results[0].skill_name == "escalation_detection"

    def test_get_skills_for_category_multiple_matches(self, registry):
        results = registry.get_skills_for_category("payment_issue")
        names = {r.skill_name for r in results}
        assert "ticket_triage" in names

    def test_get_skills_for_category_no_match(self, registry):
        results = registry.get_skills_for_category("nonexistent_category")
        assert results == []

    def test_get_skills_for_category_guest_communication(self, registry):
        results = registry.get_skills_for_category("guest_communication")
        assert len(results) == 1
        assert results[0].skill_name == "guest_response"

    def test_validate_context_valid(self, registry):
        context = {"conversation_id": "c1", "conversation_text": "Hello"}
        is_valid, missing = registry.validate_context("ticket_triage", context)
        assert is_valid is True
        assert missing == []

    def test_validate_context_missing_fields(self, registry):
        context = {"conversation_id": "c1"}
        is_valid, missing = registry.validate_context("ticket_triage", context)
        assert is_valid is False
        assert "conversation_text" in missing

    def test_validate_context_empty_context(self, registry):
        is_valid, missing = registry.validate_context("ticket_triage", {})
        assert is_valid is False
        assert len(missing) == 2

    def test_validate_context_extra_fields_ok(self, registry):
        context = {
            "conversation_id": "c1",
            "conversation_text": "hello",
            "extra_field": "ok",
        }
        is_valid, missing = registry.validate_context("ticket_triage", context)
        assert is_valid is True

    def test_validate_context_unknown_skill_raises(self, registry):
        with pytest.raises(KeyError):
            registry.validate_context("unknown", {})

    def test_validate_context_refund_handling(self, registry):
        context = {
            "conversation_id": "c1",
            "conversation_text": "refund please",
            "booking_id": "b1",
            "booking_value": 100.0,
        }
        is_valid, missing = registry.validate_context("refund_handling", context)
        assert is_valid is True
        assert missing == []

    def test_validate_context_refund_handling_partial(self, registry):
        context = {"conversation_id": "c1", "conversation_text": "refund"}
        is_valid, missing = registry.validate_context("refund_handling", context)
        assert is_valid is False
        assert "booking_id" in missing
        assert "booking_value" in missing

    def test_empty_registry(self):
        reg = WOSSkillRegistry()
        assert len(reg) == 0
        assert reg.list_skills() == []

    def test_none_skills_argument(self):
        reg = WOSSkillRegistry(skills=None)
        assert len(reg) == 0

    def test_explicit_skills_list(self):
        cfg = WOSSkillConfig(
            skill_name="custom",
            description="custom skill",
            required_context_fields=["a"],
            output_fields=["b"],
            autonomy_threshold=0.5,
            risk_level=RiskLevel.low,
            os_task_categories=["cat"],
            match_fields=["b"],
        )
        reg = WOSSkillRegistry(skills=[cfg])
        assert len(reg) == 1
        assert "custom" in reg


# ============================================================================
# Autonomy Gating
# ============================================================================


class TestAutonomyGating:
    @pytest.fixture
    def registry(self):
        return create_skill_registry()

    def test_low_risk_high_confidence_true(self, registry):
        assert registry.can_operate_autonomously("ticket_triage", 0.90) is True

    def test_low_risk_at_threshold_true(self, registry):
        assert registry.can_operate_autonomously("ticket_triage", 0.85) is True

    def test_low_risk_below_threshold_false(self, registry):
        assert registry.can_operate_autonomously("ticket_triage", 0.84) is False

    def test_low_risk_zero_confidence_false(self, registry):
        assert registry.can_operate_autonomously("ticket_triage", 0.0) is False

    def test_high_risk_always_false(self, registry):
        assert registry.can_operate_autonomously("guest_response", 1.0) is False

    def test_high_risk_refund_always_false(self, registry):
        assert registry.can_operate_autonomously("refund_handling", 1.0) is False

    def test_medium_risk_always_false(self, registry):
        assert registry.can_operate_autonomously("booking_modification", 1.0) is False

    def test_escalation_low_risk_high_confidence(self, registry):
        assert registry.can_operate_autonomously("escalation_detection", 0.85) is True

    def test_escalation_low_risk_at_threshold(self, registry):
        assert registry.can_operate_autonomously("escalation_detection", 0.80) is True

    def test_escalation_low_risk_below_threshold(self, registry):
        assert registry.can_operate_autonomously("escalation_detection", 0.79) is False

    def test_unknown_skill_raises(self, registry):
        with pytest.raises(KeyError):
            registry.can_operate_autonomously("unknown", 1.0)


# ============================================================================
# TriageEvaluator
# ============================================================================


class TestTriageEvaluator:
    def setup_method(self):
        self.evaluator = TriageEvaluator()
        self.config = _triage_config()

    def test_perfect_match_score_one(self):
        fields = {"category": "complaint", "priority": "high", "suggested_assignee": "Alice"}
        local = _make_response(fields)
        remote = _make_response(fields)
        result = self.evaluator.evaluate(local, remote, self.config)
        assert abs(result.score - 1.0) < 1e-9

    def test_total_mismatch_score_zero(self):
        local = _make_response({"category": "a", "priority": "b", "suggested_assignee": "c"})
        remote = _make_response({"category": "x", "priority": "y", "suggested_assignee": "z"})
        result = self.evaluator.evaluate(local, remote, self.config)
        assert abs(result.score) < 1e-9

    def test_category_only_match(self):
        local = _make_response({"category": "complaint", "priority": "low", "suggested_assignee": "Bob"})
        remote = _make_response({"category": "complaint", "priority": "high", "suggested_assignee": "Alice"})
        result = self.evaluator.evaluate(local, remote, self.config)
        assert abs(result.score - 0.50) < 1e-9

    def test_priority_only_match(self):
        local = _make_response({"category": "x", "priority": "high", "suggested_assignee": "Bob"})
        remote = _make_response({"category": "y", "priority": "high", "suggested_assignee": "Alice"})
        result = self.evaluator.evaluate(local, remote, self.config)
        assert abs(result.score - 0.30) < 1e-9

    def test_assignee_only_match(self):
        local = _make_response({"category": "x", "priority": "z", "suggested_assignee": "Alice"})
        remote = _make_response({"category": "y", "priority": "w", "suggested_assignee": "Alice"})
        result = self.evaluator.evaluate(local, remote, self.config)
        assert abs(result.score - 0.20) < 1e-9

    def test_category_and_priority_match(self):
        local = _make_response({"category": "complaint", "priority": "high", "suggested_assignee": "Bob"})
        remote = _make_response({"category": "complaint", "priority": "high", "suggested_assignee": "Alice"})
        result = self.evaluator.evaluate(local, remote, self.config)
        assert abs(result.score - 0.80) < 1e-9

    def test_missing_field_treated_as_none(self):
        local = _make_response({"category": "complaint"})
        remote = _make_response({"category": "complaint", "priority": "high", "suggested_assignee": "Alice"})
        result = self.evaluator.evaluate(local, remote, self.config)
        # category matches (0.5), priority/assignee miss
        assert abs(result.score - 0.50) < 1e-9

    def test_both_missing_field_match_as_none(self):
        local = _make_response({})
        remote = _make_response({})
        result = self.evaluator.evaluate(local, remote, self.config)
        # All None == None -> match
        assert abs(result.score - 1.0) < 1e-9

    def test_field_breakdown_keys(self):
        fields = {"category": "a", "priority": "b", "suggested_assignee": "c"}
        result = self.evaluator.evaluate(
            _make_response(fields), _make_response(fields), self.config
        )
        assert set(result.field_breakdown.keys()) == {"category", "priority", "suggested_assignee"}

    def test_field_breakdown_values_type(self):
        fields = {"category": "a", "priority": "b", "suggested_assignee": "c"}
        result = self.evaluator.evaluate(
            _make_response(fields), _make_response(fields), self.config
        )
        for fe in result.field_breakdown.values():
            assert isinstance(fe, FieldEvaluation)

    def test_evaluator_type_is_triage(self):
        fields = {"category": "a", "priority": "b", "suggested_assignee": "c"}
        result = self.evaluator.evaluate(
            _make_response(fields), _make_response(fields), self.config
        )
        assert result.evaluator_type == "triage"

    def test_score_always_clamped(self):
        result = self.evaluator.evaluate(
            _make_response({}), _make_response({}), self.config
        )
        assert 0.0 <= result.score <= 1.0

    def test_timestamp_present(self):
        result = self.evaluator.evaluate(
            _make_response({}), _make_response({}), self.config
        )
        assert result.timestamp is not None
        assert len(result.timestamp) > 0

    def test_error_is_none(self):
        result = self.evaluator.evaluate(
            _make_response({}), _make_response({}), self.config
        )
        assert result.error is None

    def test_satisfies_evaluator_protocol(self):
        assert isinstance(self.evaluator, EvaluatorProtocol)


# ============================================================================
# RefundEvaluator
# ============================================================================


class TestRefundEvaluator:
    def setup_method(self):
        self.evaluator = RefundEvaluator()
        self.config = _refund_config()

    def test_perfect_match(self):
        fields = {"action": "approve", "amount": 100.0, "category": "policy"}
        result = self.evaluator.evaluate(
            _make_response(fields), _make_response(fields), self.config
        )
        assert abs(result.score - 1.0) < 1e-9

    def test_total_mismatch(self):
        local = _make_response({"action": "approve", "amount": 100.0, "category": "policy"})
        remote = _make_response({"action": "deny", "amount": 999.0, "category": "other"})
        result = self.evaluator.evaluate(local, remote, self.config)
        assert abs(result.score) < 1e-9

    def test_amount_within_tolerance(self):
        local = _make_response({"action": "approve", "amount": 100.0, "category": "policy"})
        remote = _make_response({"action": "approve", "amount": 104.0, "category": "policy"})
        result = self.evaluator.evaluate(local, remote, self.config)
        # amount within 5% (4/104 ~ 3.8%), action and category match
        assert abs(result.score - 1.0) < 1e-9

    def test_amount_at_tolerance_boundary(self):
        local = _make_response({"action": "approve", "amount": 100.0, "category": "policy"})
        remote = _make_response({"action": "approve", "amount": 105.0, "category": "policy"})
        # |100-105|/105 = 5/105 ~ 4.76% <= 5% -> match
        result = self.evaluator.evaluate(local, remote, self.config)
        assert abs(result.score - 1.0) < 1e-9

    def test_amount_outside_tolerance(self):
        local = _make_response({"action": "approve", "amount": 100.0, "category": "policy"})
        remote = _make_response({"action": "approve", "amount": 200.0, "category": "policy"})
        result = self.evaluator.evaluate(local, remote, self.config)
        # amount miss (0.4 weight), action match (0.4), category match (0.2) => 0.6
        assert abs(result.score - 0.60) < 1e-9

    def test_wrong_action_only(self):
        local = _make_response({"action": "deny", "amount": 100.0, "category": "policy"})
        remote = _make_response({"action": "approve", "amount": 100.0, "category": "policy"})
        result = self.evaluator.evaluate(local, remote, self.config)
        # action miss (0.4), amount match (0.4), category match (0.2) => 0.6
        assert abs(result.score - 0.60) < 1e-9

    def test_wrong_category_only(self):
        local = _make_response({"action": "approve", "amount": 100.0, "category": "wrong"})
        remote = _make_response({"action": "approve", "amount": 100.0, "category": "policy"})
        result = self.evaluator.evaluate(local, remote, self.config)
        # category miss (0.2), action match (0.4), amount match (0.4) => 0.8
        assert abs(result.score - 0.80) < 1e-9

    def test_amount_zero_remote(self):
        local = _make_response({"action": "approve", "amount": 0.0, "category": "policy"})
        remote = _make_response({"action": "approve", "amount": 0.0, "category": "policy"})
        result = self.evaluator.evaluate(local, remote, self.config)
        # |0-0|/max(0,0.01) = 0 <= 0.05 -> match
        assert abs(result.score - 1.0) < 1e-9

    def test_amount_near_zero_tolerance(self):
        local = _make_response({"action": "approve", "amount": 0.0004, "category": "policy"})
        remote = _make_response({"action": "approve", "amount": 0.0, "category": "policy"})
        # |0.0004 - 0| / max(0, 0.01) = 0.0004/0.01 = 0.04 <= 0.05 -> match
        result = self.evaluator.evaluate(local, remote, self.config)
        assert abs(result.score - 1.0) < 1e-9

    def test_amount_non_numeric_mismatch(self):
        local = _make_response({"action": "approve", "amount": "not_a_number", "category": "policy"})
        remote = _make_response({"action": "approve", "amount": 100.0, "category": "policy"})
        result = self.evaluator.evaluate(local, remote, self.config)
        # amount miss, action match, category match => 0.6
        assert abs(result.score - 0.60) < 1e-9

    def test_field_breakdown_keys(self):
        fields = {"action": "approve", "amount": 50.0, "category": "cat"}
        result = self.evaluator.evaluate(
            _make_response(fields), _make_response(fields), self.config
        )
        assert set(result.field_breakdown.keys()) == {"action", "amount", "category"}

    def test_evaluator_type_is_refund(self):
        fields = {"action": "approve", "amount": 50.0, "category": "cat"}
        result = self.evaluator.evaluate(
            _make_response(fields), _make_response(fields), self.config
        )
        assert result.evaluator_type == "refund"

    def test_score_clamped(self):
        result = self.evaluator.evaluate(
            _make_response({}), _make_response({}), self.config
        )
        assert 0.0 <= result.score <= 1.0

    def test_error_is_none(self):
        result = self.evaluator.evaluate(
            _make_response({}), _make_response({}), self.config
        )
        assert result.error is None

    def test_satisfies_evaluator_protocol(self):
        assert isinstance(self.evaluator, EvaluatorProtocol)

    def test_amount_none_both_sides(self):
        local = _make_response({"action": "approve", "category": "policy"})
        remote = _make_response({"action": "approve", "category": "policy"})
        result = self.evaluator.evaluate(local, remote, self.config)
        # amount: None vs None -> _amount_match returns False (can't convert to float)
        # action match (0.4), category match (0.2) => 0.6
        assert abs(result.score - 0.60) < 1e-9

    def test_integer_amounts(self):
        local = _make_response({"action": "approve", "amount": 100, "category": "policy"})
        remote = _make_response({"action": "approve", "amount": 100, "category": "policy"})
        result = self.evaluator.evaluate(local, remote, self.config)
        assert abs(result.score - 1.0) < 1e-9

    def test_string_numeric_amounts(self):
        local = _make_response({"action": "approve", "amount": "100.0", "category": "policy"})
        remote = _make_response({"action": "approve", "amount": 100.0, "category": "policy"})
        result = self.evaluator.evaluate(local, remote, self.config)
        # "100.0" is convertible to float -> match
        assert abs(result.score - 1.0) < 1e-9


# ============================================================================
# Factory function
# ============================================================================


class TestCreateSkillRegistry:
    def test_returns_registry(self):
        reg = create_skill_registry()
        assert isinstance(reg, WOSSkillRegistry)

    def test_has_five_skills(self):
        reg = create_skill_registry()
        assert len(reg) == 5

    def test_all_skills_present(self):
        reg = create_skill_registry()
        for name in [
            "ticket_triage",
            "guest_response",
            "refund_handling",
            "booking_modification",
            "escalation_detection",
        ]:
            assert name in reg

    def test_returns_new_instance_each_call(self):
        r1 = create_skill_registry()
        r2 = create_skill_registry()
        assert r1 is not r2


# ============================================================================
# EvaluationResult structure checks
# ============================================================================


class TestEvaluationResultStructure:
    def test_triage_result_has_timestamp(self):
        ev = TriageEvaluator()
        result = ev.evaluate(
            _make_response({"category": "a"}),
            _make_response({"category": "a"}),
            _triage_config(),
        )
        assert isinstance(result, EvaluationResult)
        assert isinstance(result.timestamp, str)

    def test_refund_result_has_timestamp(self):
        ev = RefundEvaluator()
        result = ev.evaluate(
            _make_response({"action": "a", "amount": 1.0, "category": "c"}),
            _make_response({"action": "a", "amount": 1.0, "category": "c"}),
            _refund_config(),
        )
        assert isinstance(result, EvaluationResult)
        assert isinstance(result.timestamp, str)

    def test_triage_result_is_frozen(self):
        ev = TriageEvaluator()
        result = ev.evaluate(
            _make_response({}), _make_response({}), _triage_config()
        )
        with pytest.raises((AttributeError, ValidationError)):
            result.score = 0.99

    def test_refund_result_is_frozen(self):
        ev = RefundEvaluator()
        result = ev.evaluate(
            _make_response({}), _make_response({}), _refund_config()
        )
        with pytest.raises((AttributeError, ValidationError)):
            result.score = 0.99

    def test_triage_field_evaluation_similarity(self):
        fields = {"category": "a", "priority": "b", "suggested_assignee": "c"}
        ev = TriageEvaluator()
        result = ev.evaluate(
            _make_response(fields), _make_response(fields), _triage_config()
        )
        for fe in result.field_breakdown.values():
            assert fe.similarity == 1.0
            assert fe.matched is True

    def test_refund_field_evaluation_mismatch_similarity(self):
        ev = RefundEvaluator()
        local = _make_response({"action": "approve", "amount": 100.0, "category": "policy"})
        remote = _make_response({"action": "deny", "amount": 999.0, "category": "other"})
        result = ev.evaluate(local, remote, _refund_config())
        for fe in result.field_breakdown.values():
            assert fe.similarity == 0.0
            assert fe.matched is False


# ============================================================================
# Cross-cutting / edge cases
# ============================================================================


class TestEdgeCases:
    def test_registry_contains_dunder(self):
        reg = create_skill_registry()
        assert ("ticket_triage" in reg) is True
        assert ("nonexistent" in reg) is False

    def test_registry_len_dunder(self):
        reg = create_skill_registry()
        assert len(reg) == 5

    def test_category_refund_returns_refund_handling(self):
        reg = create_skill_registry()
        results = reg.get_skills_for_category("refund")
        names = [r.skill_name for r in results]
        assert "refund_handling" in names

    def test_category_cancellation_returns_booking_modification(self):
        reg = create_skill_registry()
        results = reg.get_skills_for_category("cancellation")
        names = [r.skill_name for r in results]
        assert "booking_modification" in names

    def test_triage_evaluator_weights_sum_to_one(self):
        total = sum(TriageEvaluator._WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_refund_evaluator_weights_sum_to_one(self):
        total = sum(RefundEvaluator._WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_skill_configs_idempotent(self):
        c1 = skill_configs()
        c2 = skill_configs()
        assert len(c1) == len(c2)
        for a, b in zip(c1, c2):
            assert a.skill_name == b.skill_name
            assert a.autonomy_threshold == b.autonomy_threshold
