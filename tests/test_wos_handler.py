"""Unit tests for WOS handler functions."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from apprentice.wos_models import (
    EVENT_TYPE_TO_TASK,
    FEEDBACK_SCORES,
    FeedbackRecord,
    FeedbackType,
    RecommendRequest,
    SkillName,
    WOSEvent,
    WOSEventType,
)
from apprentice.wos_handler import (
    handle_event,
    handle_feedback,
    handle_recommend,
    list_skills,
)


# ---------------------------------------------------------------------------
# handle_event tests
# ---------------------------------------------------------------------------


class TestHandleEvent:
    """Tests for handle_event mapping and storage."""

    @pytest.fixture
    def mock_apprentice(self):
        a = MagicMock()
        a._audit_log = MagicMock()
        a._audit_log.log = AsyncMock()
        a._training_data_store = MagicMock()
        a._training_data_store.add_example = MagicMock()
        return a

    @pytest.fixture
    def sample_event(self):
        return WOSEvent(
            event_type=WOSEventType.message_sent,
            conversation_id="conv-123",
            agent_id="agent-1",
            organization_id="org-1",
            timestamp="2024-01-01T00:00:00Z",
            payload={"message_id": "msg-1", "text": "hello"},
        )

    async def test_event_returns_accepted(self, mock_apprentice, sample_event):
        result = await handle_event(mock_apprentice, sample_event)
        assert result.status == "accepted"

    async def test_event_stores_training_data(self, mock_apprentice, sample_event):
        await handle_event(mock_apprentice, sample_event)
        mock_apprentice._training_data_store.add_example.assert_called_once()

    async def test_event_logs_audit(self, mock_apprentice, sample_event):
        await handle_event(mock_apprentice, sample_event)
        mock_apprentice._audit_log.log.assert_called_once()

    async def test_event_maps_message_to_guest_response(self, mock_apprentice, sample_event):
        """message_sent should map to guest_response task."""
        await handle_event(mock_apprentice, sample_event)
        call_args = mock_apprentice._training_data_store.add_example.call_args
        example = call_args[0][0]
        assert example.task_type == "guest_response"

    async def test_event_maps_refund_to_refund_handling(self, mock_apprentice):
        event = WOSEvent(
            event_type=WOSEventType.refund_requested,
            conversation_id="inv-456",
            timestamp="2024-01-01T00:00:00Z",
        )
        await handle_event(mock_apprentice, event)
        call_args = mock_apprentice._training_data_store.add_example.call_args
        example = call_args[0][0]
        assert example.task_type == "refund_handling"

    async def test_event_fire_and_forget_on_error(self):
        """Events should return accepted even if handler errors internally."""
        broken_apprentice = MagicMock()
        broken_apprentice._audit_log = MagicMock()
        broken_apprentice._audit_log.log = AsyncMock(side_effect=RuntimeError("boom"))
        broken_apprentice._training_data_store = None

        event = WOSEvent(
            event_type=WOSEventType.task_created,
            conversation_id="conv-1",
            timestamp="2024-01-01T00:00:00Z",
        )
        result = await handle_event(broken_apprentice, event)
        assert result.status == "accepted"

    @pytest.mark.parametrize("event_type,expected_task", list(EVENT_TYPE_TO_TASK.items()))
    async def test_all_event_type_mappings(self, mock_apprentice, event_type, expected_task):
        event = WOSEvent(
            event_type=WOSEventType(event_type),
            conversation_id="conv-1",
            timestamp="2024-01-01T00:00:00Z",
        )
        await handle_event(mock_apprentice, event)
        call_args = mock_apprentice._training_data_store.add_example.call_args
        example = call_args[0][0]
        assert example.task_type == expected_task


# ---------------------------------------------------------------------------
# handle_feedback tests
# ---------------------------------------------------------------------------


class TestHandleFeedback:
    """Tests for handle_feedback scoring logic."""

    @pytest.fixture
    def mock_apprentice(self):
        a = MagicMock()
        a._confidence_engine = MagicMock()
        a._confidence_engine.record_comparison = MagicMock()
        return a

    @pytest.fixture
    def sample_feedback(self):
        return FeedbackRecord(
            request_id="req-1",
            skill=SkillName.guest_response,
            feedback_type=FeedbackType.accept,
        )

    async def test_feedback_returns_recorded(self, mock_apprentice, sample_feedback):
        result = await handle_feedback(mock_apprentice, sample_feedback)
        assert result.status == "recorded"

    async def test_accept_score_is_1(self, mock_apprentice, sample_feedback):
        result = await handle_feedback(mock_apprentice, sample_feedback)
        assert result.match_score == 1.0

    async def test_reject_score_is_0(self, mock_apprentice):
        feedback = FeedbackRecord(
            request_id="req-2",
            skill=SkillName.refund_handling,
            feedback_type=FeedbackType.reject,
        )
        result = await handle_feedback(mock_apprentice, feedback)
        assert result.match_score == 0.0

    async def test_edit_score_is_half(self, mock_apprentice):
        feedback = FeedbackRecord(
            request_id="req-3",
            skill=SkillName.ticket_triage,
            feedback_type=FeedbackType.edit,
            edited_output={"text": "revised response"},
        )
        result = await handle_feedback(mock_apprentice, feedback)
        assert result.match_score == 0.5

    async def test_feedback_calls_confidence_engine(self, mock_apprentice, sample_feedback):
        await handle_feedback(mock_apprentice, sample_feedback)
        mock_apprentice._confidence_engine.record_comparison.assert_called_once()


# ---------------------------------------------------------------------------
# handle_recommend tests
# ---------------------------------------------------------------------------


class TestHandleRecommend:
    """Tests for handle_recommend delegation."""

    @pytest.fixture
    def mock_apprentice(self):
        a = MagicMock()
        response = MagicMock()
        response.output = {"text": "suggested response"}
        a.run = AsyncMock(return_value=response)
        a._confidence_engine = MagicMock()
        snap = MagicMock()
        snap.correlation_score = 0.85
        a._confidence_engine.get_snapshot = MagicMock(return_value=snap)
        return a

    @pytest.fixture
    def sample_request(self):
        return RecommendRequest(
            skill=SkillName.guest_response,
            context={"conversation_id": "conv-1", "message": "hello"},
            request_id="req-1",
        )

    async def test_recommend_returns_result(self, mock_apprentice, sample_request):
        result = await handle_recommend(mock_apprentice, sample_request)
        assert result.skill_name == "guest_response"
        assert result.recommendation_id == "req-1"
        assert result.confidence == 0.85
        assert result.output == {"text": "suggested response"}

    async def test_recommend_delegates_to_run(self, mock_apprentice, sample_request):
        await handle_recommend(mock_apprentice, sample_request)
        mock_apprentice.run.assert_called_once_with(
            "guest_response",
            {"conversation_id": "conv-1", "message": "hello"},
        )

    async def test_recommend_generates_request_id(self, mock_apprentice):
        request = RecommendRequest(
            skill=SkillName.refund_handling,
            context={},
        )
        result = await handle_recommend(mock_apprentice, request)
        assert result.recommendation_id  # Should be a UUID
        assert len(result.recommendation_id) > 0

    async def test_recommend_propagates_run_error(self, mock_apprentice, sample_request):
        mock_apprentice.run = AsyncMock(side_effect=RuntimeError("model unavailable"))
        with pytest.raises(RuntimeError, match="model unavailable"):
            await handle_recommend(mock_apprentice, sample_request)


# ---------------------------------------------------------------------------
# list_skills tests
# ---------------------------------------------------------------------------


class TestListSkills:
    """Tests for list_skills."""

    async def test_list_skills_empty_registry(self):
        a = MagicMock()
        a._task_registry = None
        result = await list_skills(a)
        assert result == []

    async def test_list_skills_with_tasks(self):
        a = MagicMock()
        a._task_registry = MagicMock()
        a._task_registry.task_names = {"guest_response", "refund_handling"}
        a._confidence_engine = MagicMock()
        snap = MagicMock()
        snap.phase = "COACHING"
        snap.correlation_score = 0.7
        a._confidence_engine.get_snapshot = MagicMock(return_value=snap)

        result = await list_skills(a)
        assert len(result) == 2
        names = {s.name for s in result}
        assert names == {"guest_response", "refund_handling"}
