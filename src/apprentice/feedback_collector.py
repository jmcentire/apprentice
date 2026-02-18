"""
Feedback Collector (feedback_collector) v1

Human-in-the-loop and AI-scored feedback collection for the Apprentice system.
Records accept/reject/edit/ignore/ai_score feedback per task as append-only
JSON-lines files, computes summaries and confidence adjustments from feedback history.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

_PACT_KEY = "PACT:feedback_collector"
logger = logging.getLogger(__name__)


def _log(level: str, msg: str, **kwargs) -> None:
    """Log with PACT key embedded for production traceability."""
    getattr(logger, level)(f"[{_PACT_KEY}] {msg}", **kwargs)


# ============================================================================
# ENUMS
# ============================================================================

class FeedbackType(str, Enum):
    """StrEnum representing the kind of feedback provided."""
    accept = "accept"
    reject = "reject"
    edit = "edit"
    ignore = "ignore"
    ai_score = "ai_score"


# ============================================================================
# DATA MODELS
# ============================================================================

class FeedbackEntry(BaseModel):
    """Frozen Pydantic v2 model representing a single feedback record."""
    model_config = ConfigDict(frozen=True)

    feedback_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str
    task_name: str
    feedback_type: FeedbackType
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    edited_output: Optional[dict] = None
    reason: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class FeedbackSummary(BaseModel):
    """Frozen Pydantic v2 model summarizing all feedback for a single task."""
    model_config = ConfigDict(frozen=True)

    task_name: str
    accept_count: int = 0
    reject_count: int = 0
    edit_count: int = 0
    ignore_count: int = 0
    ai_score_count: int = 0
    total_count: int = 0
    acceptance_rate: float = 0.0
    average_ai_score: float = 0.0


class FeedbackConfig(BaseModel):
    """Frozen Pydantic v2 configuration model for the FeedbackCollector."""
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    storage_dir: str = ".apprentice/feedback/"


# ============================================================================
# FEEDBACK COLLECTOR
# ============================================================================

class FeedbackCollector:
    """
    Collects, persists, and summarizes human and AI feedback per task.

    Feedback is stored as append-only JSON-lines files, one file per task,
    inside a configurable storage directory.  When disabled, all write
    operations are no-ops and read operations return empty/default values.
    """

    def __init__(
        self,
        storage_dir: str = ".apprentice/feedback/",
        enabled: bool = False,
    ) -> None:
        self._storage_dir = Path(storage_dir)
        self._enabled = enabled
        _log("info", f"FeedbackCollector initialized (enabled={enabled}, dir={storage_dir})")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _task_file(self, task_name: str) -> Path:
        """Return the JSON-lines file path for a given task."""
        return self._storage_dir / f"{task_name}.jsonl"

    def _ensure_dir(self) -> None:
        """Create the storage directory tree if it does not already exist."""
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def _read_entries(self, task_name: str) -> list[FeedbackEntry]:
        """Read all FeedbackEntry records from a task's JSON-lines file."""
        path = self._task_file(task_name)
        if not path.exists():
            return []

        entries: list[FeedbackEntry] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entries.append(FeedbackEntry(**data))
                except (json.JSONDecodeError, Exception) as exc:
                    _log("warning", f"Skipping malformed line in {path}: {exc}")
        return entries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_feedback(self, entry: FeedbackEntry) -> None:
        """
        Persist a single FeedbackEntry by appending it to the task's
        JSON-lines file.  No-op when the collector is disabled.
        """
        if not self._enabled:
            return

        self._ensure_dir()
        path = self._task_file(entry.task_name)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")

        _log("info", f"Recorded {entry.feedback_type.value} feedback for task '{entry.task_name}'")

    def get_feedback_summary(self, task_name: str) -> FeedbackSummary:
        """
        Read all feedback for *task_name* and compute an aggregate summary.
        Returns an empty summary when disabled or when no feedback exists.
        """
        if not self._enabled:
            return FeedbackSummary(task_name=task_name)

        entries = self._read_entries(task_name)
        if not entries:
            return FeedbackSummary(task_name=task_name)

        accept_count = sum(1 for e in entries if e.feedback_type == FeedbackType.accept)
        reject_count = sum(1 for e in entries if e.feedback_type == FeedbackType.reject)
        edit_count = sum(1 for e in entries if e.feedback_type == FeedbackType.edit)
        ignore_count = sum(1 for e in entries if e.feedback_type == FeedbackType.ignore)
        ai_score_count = sum(1 for e in entries if e.feedback_type == FeedbackType.ai_score)
        total_count = len(entries)

        # Acceptance rate: accepts / (accepts + rejects), 0.0 if denominator is zero
        accept_reject_total = accept_count + reject_count
        acceptance_rate = (accept_count / accept_reject_total) if accept_reject_total > 0 else 0.0

        # Average AI score: mean of score field across ai_score entries
        ai_entries = [e for e in entries if e.feedback_type == FeedbackType.ai_score]
        average_ai_score = (
            sum(e.score for e in ai_entries) / len(ai_entries)
            if ai_entries
            else 0.0
        )

        return FeedbackSummary(
            task_name=task_name,
            accept_count=accept_count,
            reject_count=reject_count,
            edit_count=edit_count,
            ignore_count=ignore_count,
            ai_score_count=ai_score_count,
            total_count=total_count,
            acceptance_rate=acceptance_rate,
            average_ai_score=average_ai_score,
        )

    def compute_feedback_adjustment(self, task_name: str) -> float:
        """
        Derive a confidence adjustment in the range [-0.1, +0.1] from the
        accept/reject ratio for *task_name*.

        * Pure accepts  -> +0.1
        * Pure rejects  -> -0.1
        * No data or disabled -> 0.0

        The value scales linearly between -0.1 and +0.1 based on
        (accepts - rejects) / (accepts + rejects).
        """
        if not self._enabled:
            return 0.0

        summary = self.get_feedback_summary(task_name)
        accept_reject_total = summary.accept_count + summary.reject_count
        if accept_reject_total == 0:
            return 0.0

        # Ratio in [-1.0, +1.0]
        ratio = (summary.accept_count - summary.reject_count) / accept_reject_total
        # Scale to [-0.1, +0.1]
        adjustment = ratio * 0.1
        # Clamp for safety
        return max(-0.1, min(0.1, adjustment))

    def list_tasks(self) -> list[str]:
        """
        Return the names of all tasks that have at least one feedback entry
        on disk.  Returns an empty list when disabled or when the storage
        directory does not exist.
        """
        if not self._enabled:
            return []

        if not self._storage_dir.exists():
            return []

        tasks: list[str] = []
        for path in sorted(self._storage_dir.iterdir()):
            if path.suffix == ".jsonl" and path.stat().st_size > 0:
                tasks.append(path.stem)
        return tasks


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "FeedbackType",
    "FeedbackEntry",
    "FeedbackSummary",
    "FeedbackConfig",
    "FeedbackCollector",
]
