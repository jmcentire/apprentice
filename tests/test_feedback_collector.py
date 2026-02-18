"""
Contract tests for FeedbackCollector.

Tests verify FeedbackEntry/FeedbackSummary/FeedbackConfig model construction,
FeedbackCollector disabled mode, record/retrieve/summarize workflows,
acceptance rate computation, confidence adjustment, task listing,
and on-disk persistence via JSON-lines.
"""

import json
import uuid

import pytest

from src.feedback_collector import (
    FeedbackType,
    FeedbackEntry,
    FeedbackSummary,
    FeedbackConfig,
    FeedbackCollector,
)


# ===================================================================
# 1. FEEDBACKENTRY MODEL TESTS
# ===================================================================

class TestFeedbackEntry:
    """Test FeedbackEntry construction, defaults, and immutability."""

    def test_creation_with_defaults(self):
        """FeedbackEntry populates feedback_id, score, and timestamp automatically."""
        entry = FeedbackEntry(
            request_id="req-001",
            task_name="summarize",
            feedback_type=FeedbackType.accept,
        )

        assert entry.request_id == "req-001"
        assert entry.task_name == "summarize"
        assert entry.feedback_type == FeedbackType.accept
        assert entry.score == 0.0
        assert entry.edited_output is None
        assert entry.reason is None
        # feedback_id should be a valid UUID string
        uuid.UUID(entry.feedback_id)
        # timestamp should be a non-empty ISO-8601 string
        assert len(entry.timestamp) > 0

    def test_creation_with_all_fields(self):
        """FeedbackEntry accepts all explicit fields."""
        entry = FeedbackEntry(
            feedback_id="custom-id-123",
            request_id="req-002",
            task_name="classify",
            feedback_type=FeedbackType.edit,
            score=0.75,
            edited_output={"label": "positive"},
            reason="Corrected sentiment label",
            timestamp="2026-01-15T12:00:00+00:00",
        )

        assert entry.feedback_id == "custom-id-123"
        assert entry.request_id == "req-002"
        assert entry.task_name == "classify"
        assert entry.feedback_type == FeedbackType.edit
        assert entry.score == 0.75
        assert entry.edited_output == {"label": "positive"}
        assert entry.reason == "Corrected sentiment label"
        assert entry.timestamp == "2026-01-15T12:00:00+00:00"

    def test_frozen_immutability(self):
        """FeedbackEntry is frozen and cannot be mutated after creation."""
        entry = FeedbackEntry(
            request_id="req-003",
            task_name="extract",
            feedback_type=FeedbackType.reject,
        )
        with pytest.raises((AttributeError, TypeError, Exception)):
            entry.score = 0.5

    def test_score_lower_bound(self):
        """Score below 0.0 is rejected by the validator."""
        with pytest.raises(Exception):
            FeedbackEntry(
                request_id="req-004",
                task_name="task",
                feedback_type=FeedbackType.ai_score,
                score=-0.1,
            )

    def test_score_upper_bound(self):
        """Score above 1.0 is rejected by the validator."""
        with pytest.raises(Exception):
            FeedbackEntry(
                request_id="req-005",
                task_name="task",
                feedback_type=FeedbackType.ai_score,
                score=1.1,
            )


# ===================================================================
# 2. FEEDBACKSUMMARY MODEL TESTS
# ===================================================================

class TestFeedbackSummary:
    """Test FeedbackSummary construction and defaults."""

    def test_creation_defaults(self):
        """FeedbackSummary defaults all counters and rates to zero."""
        summary = FeedbackSummary(task_name="summarize")

        assert summary.task_name == "summarize"
        assert summary.accept_count == 0
        assert summary.reject_count == 0
        assert summary.edit_count == 0
        assert summary.ignore_count == 0
        assert summary.ai_score_count == 0
        assert summary.total_count == 0
        assert summary.acceptance_rate == 0.0
        assert summary.average_ai_score == 0.0

    def test_frozen_immutability(self):
        """FeedbackSummary is frozen and cannot be mutated."""
        summary = FeedbackSummary(task_name="classify")
        with pytest.raises((AttributeError, TypeError, Exception)):
            summary.accept_count = 99


# ===================================================================
# 3. FEEDBACKCONFIG MODEL TESTS
# ===================================================================

class TestFeedbackConfig:
    """Test FeedbackConfig construction and defaults."""

    def test_defaults(self):
        """FeedbackConfig defaults to disabled with standard storage dir."""
        cfg = FeedbackConfig()
        assert cfg.enabled is False
        assert cfg.storage_dir == ".apprentice/feedback/"

    def test_custom_values(self):
        """FeedbackConfig accepts custom values."""
        cfg = FeedbackConfig(enabled=True, storage_dir="/tmp/my_feedback/")
        assert cfg.enabled is True
        assert cfg.storage_dir == "/tmp/my_feedback/"

    def test_frozen_immutability(self):
        """FeedbackConfig is frozen and cannot be mutated."""
        cfg = FeedbackConfig()
        with pytest.raises((AttributeError, TypeError, Exception)):
            cfg.enabled = True


# ===================================================================
# 4. FEEDBACKCOLLECTOR — DISABLED MODE
# ===================================================================

class TestFeedbackCollectorDisabled:
    """When disabled, the collector is a safe no-op."""

    def test_record_feedback_noop(self, tmp_path):
        """record_feedback does nothing when disabled."""
        collector = FeedbackCollector(storage_dir=str(tmp_path / "fb"), enabled=False)
        entry = FeedbackEntry(
            request_id="req-100",
            task_name="summarize",
            feedback_type=FeedbackType.accept,
        )
        collector.record_feedback(entry)
        # Directory should not be created
        assert not (tmp_path / "fb").exists()

    def test_get_feedback_summary_empty(self, tmp_path):
        """get_feedback_summary returns empty summary when disabled."""
        collector = FeedbackCollector(storage_dir=str(tmp_path / "fb"), enabled=False)
        summary = collector.get_feedback_summary("summarize")

        assert summary.task_name == "summarize"
        assert summary.total_count == 0
        assert summary.acceptance_rate == 0.0

    def test_compute_feedback_adjustment_zero(self, tmp_path):
        """compute_feedback_adjustment returns 0.0 when disabled."""
        collector = FeedbackCollector(storage_dir=str(tmp_path / "fb"), enabled=False)
        assert collector.compute_feedback_adjustment("summarize") == 0.0

    def test_list_tasks_empty(self, tmp_path):
        """list_tasks returns empty list when disabled."""
        collector = FeedbackCollector(storage_dir=str(tmp_path / "fb"), enabled=False)
        assert collector.list_tasks() == []


# ===================================================================
# 5. FEEDBACKCOLLECTOR — RECORD AND RETRIEVE
# ===================================================================

class TestFeedbackCollectorRecordRetrieve:
    """Test record_feedback -> get_feedback_summary round-trip."""

    def test_record_and_retrieve_summary(self, tmp_path):
        """Recording a single accept entry yields correct summary."""
        collector = FeedbackCollector(storage_dir=str(tmp_path / "fb"), enabled=True)
        entry = FeedbackEntry(
            request_id="req-200",
            task_name="summarize",
            feedback_type=FeedbackType.accept,
        )
        collector.record_feedback(entry)

        summary = collector.get_feedback_summary("summarize")
        assert summary.task_name == "summarize"
        assert summary.accept_count == 1
        assert summary.total_count == 1
        assert summary.acceptance_rate == 1.0

    def test_multiple_feedback_types(self, tmp_path):
        """Summary correctly tallies mixed feedback types."""
        collector = FeedbackCollector(storage_dir=str(tmp_path / "fb"), enabled=True)

        types_to_record = [
            FeedbackType.accept,
            FeedbackType.accept,
            FeedbackType.reject,
            FeedbackType.edit,
            FeedbackType.ignore,
            FeedbackType.ai_score,
        ]
        for i, ft in enumerate(types_to_record):
            entry = FeedbackEntry(
                request_id=f"req-{300 + i}",
                task_name="classify",
                feedback_type=ft,
                score=0.8 if ft == FeedbackType.ai_score else 0.0,
            )
            collector.record_feedback(entry)

        summary = collector.get_feedback_summary("classify")
        assert summary.accept_count == 2
        assert summary.reject_count == 1
        assert summary.edit_count == 1
        assert summary.ignore_count == 1
        assert summary.ai_score_count == 1
        assert summary.total_count == 6

    def test_acceptance_rate_computation(self, tmp_path):
        """acceptance_rate = accepts / (accepts + rejects)."""
        collector = FeedbackCollector(storage_dir=str(tmp_path / "fb"), enabled=True)

        # 3 accepts, 1 reject -> rate = 3/4 = 0.75
        for ft in [FeedbackType.accept, FeedbackType.accept, FeedbackType.accept, FeedbackType.reject]:
            collector.record_feedback(FeedbackEntry(
                request_id=str(uuid.uuid4()),
                task_name="rate_task",
                feedback_type=ft,
            ))

        summary = collector.get_feedback_summary("rate_task")
        assert summary.acceptance_rate == pytest.approx(0.75)

    def test_acceptance_rate_no_rejects(self, tmp_path):
        """acceptance_rate is 1.0 when there are only accepts."""
        collector = FeedbackCollector(storage_dir=str(tmp_path / "fb"), enabled=True)

        for _ in range(5):
            collector.record_feedback(FeedbackEntry(
                request_id=str(uuid.uuid4()),
                task_name="perfect_task",
                feedback_type=FeedbackType.accept,
            ))

        summary = collector.get_feedback_summary("perfect_task")
        assert summary.acceptance_rate == pytest.approx(1.0)

    def test_acceptance_rate_no_accepts(self, tmp_path):
        """acceptance_rate is 0.0 when there are only rejects."""
        collector = FeedbackCollector(storage_dir=str(tmp_path / "fb"), enabled=True)

        for _ in range(3):
            collector.record_feedback(FeedbackEntry(
                request_id=str(uuid.uuid4()),
                task_name="bad_task",
                feedback_type=FeedbackType.reject,
            ))

        summary = collector.get_feedback_summary("bad_task")
        assert summary.acceptance_rate == pytest.approx(0.0)

    def test_average_ai_score(self, tmp_path):
        """average_ai_score is the mean of scores across ai_score entries."""
        collector = FeedbackCollector(storage_dir=str(tmp_path / "fb"), enabled=True)

        scores = [0.6, 0.8, 1.0]
        for s in scores:
            collector.record_feedback(FeedbackEntry(
                request_id=str(uuid.uuid4()),
                task_name="scored_task",
                feedback_type=FeedbackType.ai_score,
                score=s,
            ))

        summary = collector.get_feedback_summary("scored_task")
        assert summary.average_ai_score == pytest.approx(0.8)

    def test_summary_for_unknown_task(self, tmp_path):
        """get_feedback_summary for a task with no data returns empty summary."""
        collector = FeedbackCollector(storage_dir=str(tmp_path / "fb"), enabled=True)
        summary = collector.get_feedback_summary("nonexistent")
        assert summary.total_count == 0
        assert summary.acceptance_rate == 0.0


# ===================================================================
# 6. FEEDBACKCOLLECTOR — CONFIDENCE ADJUSTMENT
# ===================================================================

class TestFeedbackCollectorAdjustment:
    """Test compute_feedback_adjustment."""

    def test_positive_adjustment(self, tmp_path):
        """Mostly accepts yields a positive adjustment."""
        collector = FeedbackCollector(storage_dir=str(tmp_path / "fb"), enabled=True)

        # 9 accepts, 1 reject -> ratio = 8/10 = 0.8 -> adjustment = 0.08
        for _ in range(9):
            collector.record_feedback(FeedbackEntry(
                request_id=str(uuid.uuid4()),
                task_name="good_task",
                feedback_type=FeedbackType.accept,
            ))
        collector.record_feedback(FeedbackEntry(
            request_id=str(uuid.uuid4()),
            task_name="good_task",
            feedback_type=FeedbackType.reject,
        ))

        adj = collector.compute_feedback_adjustment("good_task")
        assert adj > 0.0
        assert adj <= 0.1

    def test_negative_adjustment(self, tmp_path):
        """Mostly rejects yields a negative adjustment."""
        collector = FeedbackCollector(storage_dir=str(tmp_path / "fb"), enabled=True)

        # 1 accept, 9 rejects -> ratio = -8/10 = -0.8 -> adjustment = -0.08
        collector.record_feedback(FeedbackEntry(
            request_id=str(uuid.uuid4()),
            task_name="bad_task",
            feedback_type=FeedbackType.accept,
        ))
        for _ in range(9):
            collector.record_feedback(FeedbackEntry(
                request_id=str(uuid.uuid4()),
                task_name="bad_task",
                feedback_type=FeedbackType.reject,
            ))

        adj = collector.compute_feedback_adjustment("bad_task")
        assert adj < 0.0
        assert adj >= -0.1

    def test_all_accepts_max_positive(self, tmp_path):
        """All accepts produces +0.1."""
        collector = FeedbackCollector(storage_dir=str(tmp_path / "fb"), enabled=True)

        for _ in range(5):
            collector.record_feedback(FeedbackEntry(
                request_id=str(uuid.uuid4()),
                task_name="perfect",
                feedback_type=FeedbackType.accept,
            ))

        adj = collector.compute_feedback_adjustment("perfect")
        assert adj == pytest.approx(0.1)

    def test_all_rejects_max_negative(self, tmp_path):
        """All rejects produces -0.1."""
        collector = FeedbackCollector(storage_dir=str(tmp_path / "fb"), enabled=True)

        for _ in range(5):
            collector.record_feedback(FeedbackEntry(
                request_id=str(uuid.uuid4()),
                task_name="terrible",
                feedback_type=FeedbackType.reject,
            ))

        adj = collector.compute_feedback_adjustment("terrible")
        assert adj == pytest.approx(-0.1)

    def test_no_feedback_returns_zero(self, tmp_path):
        """No feedback for a task yields 0.0 adjustment."""
        collector = FeedbackCollector(storage_dir=str(tmp_path / "fb"), enabled=True)
        adj = collector.compute_feedback_adjustment("empty_task")
        assert adj == 0.0

    def test_only_non_accept_reject_returns_zero(self, tmp_path):
        """Only edit/ignore/ai_score entries (no accept or reject) yields 0.0."""
        collector = FeedbackCollector(storage_dir=str(tmp_path / "fb"), enabled=True)

        for ft in [FeedbackType.edit, FeedbackType.ignore, FeedbackType.ai_score]:
            collector.record_feedback(FeedbackEntry(
                request_id=str(uuid.uuid4()),
                task_name="neutral",
                feedback_type=ft,
                score=0.5 if ft == FeedbackType.ai_score else 0.0,
            ))

        adj = collector.compute_feedback_adjustment("neutral")
        assert adj == 0.0

    def test_adjustment_bounded(self, tmp_path):
        """Adjustment is always in [-0.1, +0.1]."""
        collector = FeedbackCollector(storage_dir=str(tmp_path / "fb"), enabled=True)

        # Record a mix
        for _ in range(50):
            collector.record_feedback(FeedbackEntry(
                request_id=str(uuid.uuid4()),
                task_name="bounded",
                feedback_type=FeedbackType.accept,
            ))
        for _ in range(50):
            collector.record_feedback(FeedbackEntry(
                request_id=str(uuid.uuid4()),
                task_name="bounded",
                feedback_type=FeedbackType.reject,
            ))

        adj = collector.compute_feedback_adjustment("bounded")
        assert -0.1 <= adj <= 0.1


# ===================================================================
# 7. FEEDBACKCOLLECTOR — LIST TASKS
# ===================================================================

class TestFeedbackCollectorListTasks:
    """Test list_tasks."""

    def test_list_tasks_multiple(self, tmp_path):
        """list_tasks returns all tasks with recorded feedback."""
        collector = FeedbackCollector(storage_dir=str(tmp_path / "fb"), enabled=True)

        for task in ["alpha", "beta", "gamma"]:
            collector.record_feedback(FeedbackEntry(
                request_id=str(uuid.uuid4()),
                task_name=task,
                feedback_type=FeedbackType.accept,
            ))

        tasks = collector.list_tasks()
        assert set(tasks) == {"alpha", "beta", "gamma"}

    def test_list_tasks_empty_when_no_data(self, tmp_path):
        """list_tasks returns empty list when no feedback has been recorded."""
        collector = FeedbackCollector(storage_dir=str(tmp_path / "fb"), enabled=True)
        assert collector.list_tasks() == []

    def test_list_tasks_no_directory(self, tmp_path):
        """list_tasks returns empty when storage dir does not exist."""
        collector = FeedbackCollector(
            storage_dir=str(tmp_path / "nonexistent_fb"),
            enabled=True,
        )
        assert collector.list_tasks() == []


# ===================================================================
# 8. FEEDBACKCOLLECTOR — PERSISTENCE TO DISK
# ===================================================================

class TestFeedbackCollectorPersistence:
    """Test that data is persisted as JSON-lines on disk and survives re-instantiation."""

    def test_file_created_on_record(self, tmp_path):
        """Recording feedback creates a .jsonl file for the task."""
        storage = str(tmp_path / "fb")
        collector = FeedbackCollector(storage_dir=storage, enabled=True)

        collector.record_feedback(FeedbackEntry(
            request_id="req-persist-1",
            task_name="persist_task",
            feedback_type=FeedbackType.accept,
        ))

        jsonl_path = tmp_path / "fb" / "persist_task.jsonl"
        assert jsonl_path.exists()

    def test_jsonl_format(self, tmp_path):
        """Each line in the file is valid JSON representing a FeedbackEntry."""
        storage = str(tmp_path / "fb")
        collector = FeedbackCollector(storage_dir=storage, enabled=True)

        for i in range(3):
            collector.record_feedback(FeedbackEntry(
                request_id=f"req-fmt-{i}",
                task_name="format_task",
                feedback_type=FeedbackType.accept,
            ))

        jsonl_path = tmp_path / "fb" / "format_task.jsonl"
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 3

        for line in lines:
            data = json.loads(line)
            assert "request_id" in data
            assert "feedback_type" in data

    def test_survives_reinstantiation(self, tmp_path):
        """Data written by one collector instance is readable by a new one."""
        storage = str(tmp_path / "fb")

        collector1 = FeedbackCollector(storage_dir=storage, enabled=True)
        for i in range(5):
            collector1.record_feedback(FeedbackEntry(
                request_id=f"req-surv-{i}",
                task_name="survive_task",
                feedback_type=FeedbackType.accept if i < 3 else FeedbackType.reject,
            ))

        # New instance, same directory
        collector2 = FeedbackCollector(storage_dir=storage, enabled=True)
        summary = collector2.get_feedback_summary("survive_task")

        assert summary.total_count == 5
        assert summary.accept_count == 3
        assert summary.reject_count == 2
        assert summary.acceptance_rate == pytest.approx(0.6)

    def test_append_only_semantics(self, tmp_path):
        """Subsequent record_feedback calls append, never overwrite."""
        storage = str(tmp_path / "fb")
        collector = FeedbackCollector(storage_dir=storage, enabled=True)

        collector.record_feedback(FeedbackEntry(
            request_id="req-app-1",
            task_name="append_task",
            feedback_type=FeedbackType.accept,
        ))

        # Re-instantiate and append more
        collector2 = FeedbackCollector(storage_dir=storage, enabled=True)
        collector2.record_feedback(FeedbackEntry(
            request_id="req-app-2",
            task_name="append_task",
            feedback_type=FeedbackType.reject,
        ))

        summary = collector2.get_feedback_summary("append_task")
        assert summary.total_count == 2
        assert summary.accept_count == 1
        assert summary.reject_count == 1
