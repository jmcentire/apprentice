"""
Audit Log re-export module.

This module re-exports all audit_log functionality for the reporting composition,
adapting the event type enum to match parent contract expectations.
"""

from enum import Enum

# Import base modules from audit_log implementation (intra-package)
from .audit_log import (
    AuditConfig as _AuditConfig,
    AuditEntry as _AuditEntry,
    JsonLinesAuditLogger as _JsonLinesAuditLogger,
    AuditEntryList as _AuditEntryList,
    EventType as _EventType,
)


# Create adapter EventType enum with uppercase names to match parent contract
class EventType(str, Enum):
    """Adapter enum matching parent contract expectations (uppercase names)."""
    REQUEST_ROUTED = "request_routed"
    TRAINING_EXAMPLE_STORED = "training_example_stored"
    FINE_TUNE_STARTED = "fine_tune_started"
    FINE_TUNE_COMPLETED = "fine_tune_completed"
    MODEL_VALIDATED = "model_validated"
    MODEL_PROMOTED = "model_promoted"
    PHASE_TRANSITION = "phase_transition"
    BUDGET_WARNING = "budget_warning"
    CONFIDENCE_ALERT = "confidence_alert"


# Map parent enum to child enum
def _to_child_event_type(event_type):
    """Convert parent EventType to child EventType."""
    if isinstance(event_type, str):
        # String value, find matching child enum
        return _EventType(event_type)
    elif hasattr(event_type, 'value'):
        # Parent enum, use value
        return _EventType(event_type.value)
    else:
        return event_type


# Adapter for AuditConfig that converts event types and adds missing fields
class AuditConfig:
    """Adapter for AuditConfig that converts event types between parent and child enums."""

    def __init__(self, log_path, max_file_size_mb=100, rotation_backup_count=5, enabled_event_types=None):
        # Validate fields before creating
        if not log_path or not log_path.strip():
            raise ValueError("log_path must be a non-empty string")

        if not 1 <= max_file_size_mb <= 10000:
            raise ValueError("max_file_size_mb must be between 1 and 10000")

        if not 0 <= rotation_backup_count <= 100:
            raise ValueError("rotation_backup_count must be between 0 and 100")

        # Child AuditConfig only has log_path, so we need to store the other fields ourselves
        self._config = _AuditConfig(log_path=log_path)

        # Store the additional fields that parent contract expects but child doesn't have
        object.__setattr__(self, '_max_file_size_mb', max_file_size_mb)
        object.__setattr__(self, '_rotation_backup_count', rotation_backup_count)
        object.__setattr__(self, '_enabled_event_types', enabled_event_types or [])

    @property
    def log_path(self):
        return self._config.log_path

    @property
    def max_file_size_mb(self):
        return self._max_file_size_mb

    @property
    def rotation_backup_count(self):
        return self._rotation_backup_count

    @property
    def enabled_event_types(self):
        return self._enabled_event_types

    def __setattr__(self, name, value):
        """Prevent attribute assignment (frozen)."""
        if name in ('_config', '_max_file_size_mb', '_rotation_backup_count', '_enabled_event_types'):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(f"Cannot set attribute '{name}' on frozen instance")


# Adapter for AuditEntry that converts event types
class AuditEntry:
    """Adapter for AuditEntry that converts event types between parent and child enums."""

    def __init__(self, timestamp=None, task_name=None, event_type=None, details=None, correlation_id=""):
        # Convert parent EventType to child EventType
        child_event_type = _to_child_event_type(event_type) if event_type else None

        kwargs = {}
        if timestamp is not None:
            kwargs['timestamp'] = timestamp
        if task_name is not None:
            kwargs['task_name'] = task_name
        if child_event_type is not None:
            kwargs['event_type'] = child_event_type
        if details is not None:
            kwargs['details'] = details
        if correlation_id:
            kwargs['correlation_id'] = correlation_id

        self._entry = _AuditEntry(**kwargs)

    @property
    def timestamp(self):
        return self._entry.timestamp

    @property
    def task_name(self):
        return self._entry.task_name

    @property
    def event_type(self):
        # Convert back to parent enum
        return EventType(self._entry.event_type.value)

    @property
    def details(self):
        return self._entry.details

    @property
    def correlation_id(self):
        return self._entry.correlation_id if hasattr(self._entry, 'correlation_id') else ""

    def to_json_line(self):
        """Delegate to child entry."""
        return self._entry.to_json_line()

    def __setattr__(self, name, value):
        """Prevent attribute assignment (frozen)."""
        if name == '_entry':
            object.__setattr__(self, name, value)
        else:
            raise TypeError(f"Cannot set attribute '{name}' on frozen instance")


# Wrapper for JsonLinesAuditLogger that accepts our adapter AuditConfig
class JsonLinesAuditLogger:
    """Wrapper for JsonLinesAuditLogger that works with our adapter classes."""

    def __init__(self, config: AuditConfig):
        # Create child config from our adapter config
        child_config = config._config
        self._logger = _JsonLinesAuditLogger(child_config)
        self._enabled_types = config.enabled_event_types

    def log(self, entry: AuditEntry):
        """Log an entry if its event type is enabled."""
        # Check if event type is enabled
        if self._enabled_types and entry.event_type not in self._enabled_types:
            # Silently drop
            return

        # Convert to child entry and log
        self._logger.log(entry._entry)

    def entries_by_task(self, task_name: str) -> list:
        """Query entries by task name."""
        child_entries = self._logger.entries_by_task(task_name)
        # Wrap each child entry in our adapter
        return [_wrap_child_entry(e) for e in child_entries]

    def entries_by_type(self, event_type: EventType) -> list:
        """Query entries by event type."""
        child_event_type = _to_child_event_type(event_type)
        child_entries = self._logger.entries_by_type(child_event_type)
        return [_wrap_child_entry(e) for e in child_entries]

    def entries_in_range(self, start: str, end: str) -> list:
        """Query entries in time range."""
        child_entries = self._logger.entries_in_range(start, end)
        return [_wrap_child_entry(e) for e in child_entries]


def _wrap_child_entry(child_entry):
    """Wrap a child AuditEntry in our adapter."""
    return AuditEntry(
        timestamp=child_entry.timestamp,
        task_name=child_entry.task_name,
        event_type=EventType(child_entry.event_type.value),
        details=child_entry.details,
        correlation_id=getattr(child_entry, 'correlation_id', ""),
    )


# Type alias
AuditEntryList = list[AuditEntry]


__all__ = [
    'EventType',
    'AuditEntry',
    'AuditConfig',
    'JsonLinesAuditLogger',
    'AuditEntryList',
]
