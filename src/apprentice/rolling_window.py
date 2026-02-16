# === Rolling Comparison Window (rolling_window) v1 ===
# Maintains a fixed-size deque of ComparisonPair samples per task type for per-task quality tracking.

import logging
import os
import json
import re
import threading
from collections import deque
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from statistics import linear_regression, mean
from typing import Any, Optional, Protocol, List
from pydantic import BaseModel, Field, field_validator, ConfigDict

_PACT_KEY = "PACT:d93d91:rolling_window"
logger = logging.getLogger(__name__)


class PactFormatter(logging.Formatter):
    """Formatter that injects the PACT log key into every record."""

    def format(self, record):
        record.pact_key = _PACT_KEY
        return super().format(record)


def _log(level: str, msg: str, **kwargs) -> None:
    """Log with PACT key embedded for production traceability."""
    getattr(logger, level)(f"[{_PACT_KEY}] {msg}", **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# TYPE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════


class TrendDirection(Enum):
    """Direction of the evaluation score trend over the recent window of comparison pairs."""
    IMPROVING = "IMPROVING"
    DEGRADING = "DEGRADING"
    STABLE = "STABLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ComparisonPair(BaseModel):
    """Immutable (frozen) Pydantic v2 model representing a single comparison sample."""
    model_config = ConfigDict(frozen=True)

    timestamp: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00$")
    input_data: str = Field(..., min_length=1)
    local_response: str
    remote_response: str
    evaluation_score: float = Field(..., ge=0.0, le=1.0)


class ComparisonPairInput(BaseModel):
    """Caller-provided data for creating a ComparisonPair. Does not include timestamp."""
    input_data: str = Field(..., min_length=1)
    local_response: str
    remote_response: str
    evaluation_score: float = Field(..., ge=0.0, le=1.0)


OptionalFloat = Optional[float]
OptionalStr = Optional[str]
ComparisonPairList = List[ComparisonPair]


class WindowAggregate(BaseModel):
    """Immutable (frozen) Pydantic v2 result model containing aggregate statistics."""
    model_config = ConfigDict(frozen=True)

    mean_score: OptionalFloat
    trend_slope: OptionalFloat
    trend_direction: TrendDirection
    is_stale: bool
    sample_count: int = Field(..., ge=0)
    oldest_timestamp: OptionalStr


class WindowConfig(BaseModel):
    """Configuration for constructing a DequeRollingWindow instance."""
    window_size: int = Field(default=100, ge=1, le=10000)
    max_window_age_seconds: float = Field(default=86400.0, gt=0.0)


class StoreConfig(BaseModel):
    """Configuration for constructing a JsonFileWindowStore instance."""
    base_dir: str = Field(..., min_length=1)


# ═══════════════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class PersistenceError(Exception):
    """Raised when store I/O operations fail."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# WINDOW STORE PROTOCOL
# ═══════════════════════════════════════════════════════════════════════════


class WindowStore(Protocol):
    """Abstract protocol for persistence of ComparisonPair data."""

    def save(self, task_type: str, pairs: ComparisonPairList) -> None:
        ...

    def load(self, task_type: str) -> ComparisonPairList:
        ...


# ═══════════════════════════════════════════════════════════════════════════
# DEQUE ROLLING WINDOW
# ═══════════════════════════════════════════════════════════════════════════


class DequeRollingWindow:
    """Maintains a fixed-size deque of ComparisonPair samples with thread-safe operations."""

    def __init__(self, config: WindowConfig, store: Optional[WindowStore] = None) -> None:
        """
        Constructs a DequeRollingWindow backed by collections.deque(maxlen=window_size).
        """
        # Validate config (Pydantic will do this, but we catch and wrap)
        try:
            if not isinstance(config, WindowConfig):
                config = WindowConfig(**config.model_dump())
        except Exception as e:
            _log("error", f"Invalid WindowConfig: {e}")
            raise ValidationError(f"Invalid WindowConfig: {e}") from e

        self._config = config
        self._store = store
        self._deque: deque[ComparisonPair] = deque(maxlen=config.window_size)
        self._lock = threading.Lock()
        _log("info", f"DequeRollingWindow initialized with window_size={config.window_size}")

    def add(self, pair_input: ComparisonPairInput) -> ComparisonPair:
        """
        Adds a new ComparisonPair to the rolling window.
        """
        # Validate input (Pydantic will do this)
        try:
            if not isinstance(pair_input, ComparisonPairInput):
                pair_input = ComparisonPairInput(**pair_input.model_dump())
        except Exception as e:
            _log("error", f"Invalid ComparisonPairInput: {e}")
            raise ValidationError(f"Invalid ComparisonPairInput: {e}") from e

        # Assign timestamp
        timestamp_str = datetime.now(timezone.utc).isoformat()

        # Create ComparisonPair
        try:
            pair = ComparisonPair(
                timestamp=timestamp_str,
                input_data=pair_input.input_data,
                local_response=pair_input.local_response,
                remote_response=pair_input.remote_response,
                evaluation_score=pair_input.evaluation_score,
            )
        except Exception as e:
            _log("error", f"Failed to create ComparisonPair: {e}")
            raise ValidationError(f"Failed to create ComparisonPair: {e}") from e

        with self._lock:
            self._deque.append(pair)

        _log("debug", f"Added pair with timestamp={timestamp_str}, score={pair_input.evaluation_score}")
        return pair

    def aggregate(self, trend_window: Optional[int] = None) -> WindowAggregate:
        """
        Computes aggregate statistics over the current window contents.
        """
        if trend_window is not None and trend_window < 2:
            raise ValueError("trend_window must be >= 2 or None")

        with self._lock:
            pairs_snapshot = list(self._deque)

        sample_count = len(pairs_snapshot)

        # Empty window
        if sample_count == 0:
            return WindowAggregate(
                mean_score=None,
                trend_slope=None,
                trend_direction=TrendDirection.INSUFFICIENT_DATA,
                is_stale=True,
                sample_count=0,
                oldest_timestamp=None,
            )

        # Compute mean score
        scores = [p.evaluation_score for p in pairs_snapshot]
        mean_score = mean(scores)

        # Compute oldest timestamp
        oldest_timestamp = pairs_snapshot[0].timestamp

        # Check staleness
        oldest_dt = datetime.fromisoformat(oldest_timestamp)
        now_utc = datetime.now(timezone.utc)
        age_seconds = (now_utc - oldest_dt).total_seconds()
        is_stale = age_seconds > self._config.max_window_age_seconds

        # Compute trend
        trend_pairs = pairs_snapshot
        trend_slope = None
        trend_direction = TrendDirection.INSUFFICIENT_DATA
        
        if trend_window is not None:
            if trend_window > sample_count:
                # FIXED: When trend_window exceeds sample_count, return None slope and INSUFFICIENT_DATA
                trend_slope = None
                trend_direction = TrendDirection.INSUFFICIENT_DATA
            else:
                trend_pairs = pairs_snapshot[-trend_window:]
                if len(trend_pairs) >= 2:
                    # Use linear regression
                    trend_scores = [p.evaluation_score for p in trend_pairs]
                    x_values = list(range(len(trend_scores)))
                    slope, _ = linear_regression(x_values, trend_scores)
                    trend_slope = slope

                    if slope > 0:
                        trend_direction = TrendDirection.IMPROVING
                    elif slope < 0:
                        trend_direction = TrendDirection.DEGRADING
                    else:
                        trend_direction = TrendDirection.STABLE
        else:
            # No trend_window specified, use all pairs
            if len(trend_pairs) >= 2:
                trend_scores = [p.evaluation_score for p in trend_pairs]
                x_values = list(range(len(trend_scores)))
                slope, _ = linear_regression(x_values, trend_scores)
                trend_slope = slope

                if slope > 0:
                    trend_direction = TrendDirection.IMPROVING
                elif slope < 0:
                    trend_direction = TrendDirection.DEGRADING
                else:
                    trend_direction = TrendDirection.STABLE

        _log("debug", f"Aggregate: mean={mean_score}, slope={trend_slope}, stale={is_stale}")

        return WindowAggregate(
            mean_score=mean_score,
            trend_slope=trend_slope,
            trend_direction=trend_direction,
            is_stale=is_stale,
            sample_count=sample_count,
            oldest_timestamp=oldest_timestamp,
        )

    def check_staleness(self, max_age_seconds: Optional[float] = None) -> bool:
        """
        Checks whether the rolling window is stale.
        """
        if max_age_seconds is not None and max_age_seconds <= 0.0:
            raise ValueError("max_age_seconds must be positive or None")

        with self._lock:
            if len(self._deque) == 0:
                return True
            oldest_pair = self._deque[0]

        oldest_dt = datetime.fromisoformat(oldest_pair.timestamp)
        now_utc = datetime.now(timezone.utc)
        age_seconds = (now_utc - oldest_dt).total_seconds()

        max_age = max_age_seconds if max_age_seconds is not None else self._config.max_window_age_seconds
        return age_seconds > max_age

    def pairs(self) -> ComparisonPairList:
        """
        Returns a read-only snapshot of all ComparisonPair instances.
        """
        with self._lock:
            return list(self._deque)

    def __len__(self) -> int:
        """
        Returns the current number of ComparisonPair instances in the window.
        """
        with self._lock:
            return len(self._deque)

    def persist(self, task_type: str) -> None:
        """
        Persists the current window contents to the injected WindowStore.
        """
        if not task_type:
            raise ValueError("task_type must be non-empty")

        if self._store is None:
            return

        with self._lock:
            pairs_snapshot = list(self._deque)

        try:
            self._store.save(task_type, pairs_snapshot)
            _log("info", f"Persisted {len(pairs_snapshot)} pairs for task_type={task_type}")
        except Exception as e:
            _log("error", f"Store write error for task_type={task_type}: {e}")
            raise PersistenceError(f"Store write error: {e}") from e

    def load_from_store(self, task_type: str) -> int:
        """
        Loads persisted ComparisonPair data from the injected WindowStore.
        """
        if not task_type:
            raise ValueError("task_type must be non-empty")

        if self._store is None:
            return 0

        try:
            pairs = self._store.load(task_type)
            _log("info", f"Loaded {len(pairs)} pairs for task_type={task_type}")
        except Exception as e:
            _log("error", f"Store read error for task_type={task_type}: {e}")
            raise PersistenceError(f"Store read error: {e}") from e

        # Truncate to window_size if necessary (keep most recent)
        if len(pairs) > self._config.window_size:
            pairs = pairs[-self._config.window_size:]
            _log("debug", f"Truncated loaded pairs to window_size={self._config.window_size}")

        with self._lock:
            self._deque.clear()
            self._deque.extend(pairs)

        return len(pairs)


# ═══════════════════════════════════════════════════════════════════════════
# JSON FILE WINDOW STORE
# ═══════════════════════════════════════════════════════════════════════════


class JsonFileWindowStore:
    """Persists ComparisonPair data as JSON files in a filesystem directory."""

    def __init__(self, config: StoreConfig) -> None:
        """
        Constructs a JsonFileWindowStore that persists ComparisonPair data as JSON files.
        """
        try:
            if not isinstance(config, StoreConfig):
                config = StoreConfig(**config.model_dump())
        except Exception as e:
            _log("error", f"Invalid StoreConfig: {e}")
            raise ValidationError(f"Invalid StoreConfig: {e}") from e

        self._base_dir = Path(config.base_dir)

        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            _log("info", f"JsonFileWindowStore initialized with base_dir={self._base_dir}")
        except PermissionError as e:
            _log("error", f"Permission denied creating base_dir={self._base_dir}")
            raise PermissionError(f"Cannot create base_dir: {e}") from e
        except OSError as e:
            _log("error", f"Invalid path for base_dir={self._base_dir}: {e}")
            raise OSError(f"Invalid path: {e}") from e

    def _sanitize_task_type(self, task_type: str) -> str:
        """Sanitizes task_type to create a valid filename."""
        # Replace non-alphanumeric characters with underscore
        return re.sub(r'[^a-zA-Z0-9_-]', '_', task_type)

    def _get_file_path(self, task_type: str) -> Path:
        """Returns the file path for a given task_type."""
        sanitized = self._sanitize_task_type(task_type)
        return self._base_dir / f"{sanitized}.json"

    def save(self, task_type: str, pairs: ComparisonPairList) -> None:
        """
        Persists a sequence of ComparisonPair instances as a JSON file.
        """
        if not task_type:
            raise ValueError("task_type must be non-empty")

        file_path = self._get_file_path(task_type)

        # Serialize pairs
        try:
            pairs_data = [p.model_dump() for p in pairs]
            json_str = json.dumps(pairs_data, indent=2)
        except Exception as e:
            _log("error", f"Serialization error for task_type={task_type}: {e}")
            raise PersistenceError(f"Serialization failed: {e}") from e

        # Atomic write: write to temp file, then replace
        temp_path = file_path.with_suffix(".tmp")
        try:
            temp_path.write_text(json_str, encoding="utf-8")
            os.replace(temp_path, file_path)
            _log("info", f"Saved {len(pairs)} pairs to {file_path}")
        except PermissionError as e:
            _log("error", f"Write permission denied for {file_path}: {e}")
            raise PersistenceError(f"Write permission denied: {e}") from e
        except OSError as e:
            if "No space left on device" in str(e) or "Disk quota exceeded" in str(e):
                _log("error", f"Disk full writing to {file_path}: {e}")
                raise PersistenceError(f"Disk full: {e}") from e
            else:
                _log("error", f"OS error writing to {file_path}: {e}")
                raise PersistenceError(f"OS error: {e}") from e
        finally:
            # Clean up temp file if it still exists
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

    def load(self, task_type: str) -> ComparisonPairList:
        """
        Loads a list of ComparisonPair instances from the JSON file.
        """
        if not task_type:
            raise ValueError("task_type must be non-empty")

        file_path = self._get_file_path(task_type)

        if not file_path.exists():
            _log("debug", f"File not found for task_type={task_type}, returning empty list")
            return []

        try:
            json_str = file_path.read_text(encoding="utf-8")
        except PermissionError as e:
            _log("error", f"Read permission denied for {file_path}: {e}")
            raise PersistenceError(f"Read permission denied: {e}") from e
        except Exception as e:
            _log("error", f"Error reading {file_path}: {e}")
            raise PersistenceError(f"Error reading file: {e}") from e

        # Parse JSON
        try:
            pairs_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            _log("warning", f"Corrupt JSON in {file_path}, returning empty list: {e}")
            return []

        # Validate and create ComparisonPair objects
        pairs = []
        for i, pair_dict in enumerate(pairs_data):
            try:
                pair = ComparisonPair(**pair_dict)
                pairs.append(pair)
            except Exception as e:
                _log("warning", f"Invalid pair at index {i} in {file_path}, skipping: {e}")
                continue

        _log("info", f"Loaded {len(pairs)} pairs from {file_path}")
        return pairs


# ═══════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    'TrendDirection',
    'ComparisonPair',
    'ComparisonPairInput',
    'WindowAggregate',
    'OptionalFloat',
    'OptionalStr',
    'ComparisonPairList',
    'WindowConfig',
    'StoreConfig',
    'ValidationError',
    'PersistenceError',
    'DequeRollingWindow',
    'JsonFileWindowStore',
]
