"""Audit log component for structured JSON logging of system decisions.

Provides append-only audit logging with query capabilities for reports.
All timestamps are UTC. Uses stdlib logging with custom JSON formatter.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional, Protocol

from pydantic import BaseModel, Field, field_validator, ConfigDict

# ── LOG KEY PREAMBLE ──────────────────────────────────────────────────────────
_PACT_KEY = "PACT:04bd77:audit_log"
logger = logging.getLogger(__name__)


class PactFormatter(logging.Formatter):
    """Formatter that injects the PACT log key into every record."""

    def format(self, record):
        record.pact_key = _PACT_KEY
        return super().format(record)


def _log(level: str, msg: str, **kwargs) -> None:
    """Log with PACT key embedded for production traceability."""
    getattr(logger, level)(f"[{_PACT_KEY}] {msg}", **kwargs)


# ── EVENT TYPE ENUM ───────────────────────────────────────────────────────────
class EventType(str, Enum):
    """Closed enumeration of the 9 auditable system event types.
    
    StrEnum — each variant's value is its snake_case name.
    No catch-all variant; unrecognized event strings must be rejected at validation time.
    """
    request_routed = "request_routed"
    training_example_stored = "training_example_stored"
    fine_tune_started = "fine_tune_started"
    fine_tune_completed = "fine_tune_completed"
    model_validated = "model_validated"
    model_promoted = "model_promoted"
    phase_transition = "phase_transition"
    budget_warning = "budget_warning"
    confidence_alert = "confidence_alert"


# ── AUDIT ENTRY MODEL ─────────────────────────────────────────────────────────
class AuditEntry(BaseModel):
    """Immutable (frozen) Pydantic v2 BaseModel representing a single auditable system decision.
    
    The timestamp defaults to datetime.now(UTC) via default_factory.
    Provides .to_json_line() -> str for JSONL serialization (compact JSON, no trailing newline).
    """
    model_config = ConfigDict(frozen=True)

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00$"
    )
    task_name: str = Field(min_length=1)
    event_type: EventType
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator('timestamp', mode='before')
    @classmethod
    def validate_timestamp(cls, v: Any) -> str:
        """Convert datetime to ISO 8601 UTC string if needed."""
        if isinstance(v, datetime):
            if v.tzinfo is None:
                raise ValueError("timestamp must be timezone-aware")
            return v.isoformat()
        return v

    def to_json_line(self) -> str:
        """Serialize this AuditEntry to a single-line compact JSON string.
        
        Uses Pydantic's model_dump_json() with no indentation.
        Does not include a trailing newline character.
        The output is deterministic for a given entry (sorted keys).
        
        Returns:
            str: Compact JSON representation, no trailing newline
            
        Raises:
            ValueError: If details dict contains non-JSON-serializable value
        """
        try:
            # model_dump_json produces compact JSON by default
            json_str = self.model_dump_json()
            # Ensure no newlines (should not be present in compact JSON)
            if '\n' in json_str:
                json_str = json_str.replace('\n', '')
            return json_str
        except (TypeError, ValueError) as e:
            raise ValueError(f"Details dict contains non-JSON-serializable value: {e}")

    def __setattr__(self, name, value):
        """Override __setattr__ to enforce immutability with clear error."""
        if hasattr(self, '__pydantic_private__'):
            # Model is already initialized, reject any mutation
            raise TypeError(f"Cannot set attribute '{name}' on frozen instance")
        # During initialization, allow Pydantic to set attributes
        object.__setattr__(self, name, value)


# ── AUDIT CONFIG MODEL ────────────────────────────────────────────────────────
class AuditConfig(BaseModel):
    """Pydantic v2 BaseModel for audit logger configuration.

    Validated at startup — fail fast on invalid config.
    Typically loaded from the YAML config file's 'audit' section.
    """
    log_path: str = Field(min_length=1)
    log_level: str = "INFO"
    log_to_stdout: bool = False
    max_file_size_mb: int = 100
    backup_count: int = 5

    @field_validator('log_path')
    @classmethod
    def validate_log_path(cls, v: str) -> str:
        """Validate that log_path parent directory exists and is writable."""
        if not v or len(v.strip()) == 0:
            raise ValueError("log_path must be a non-empty string")

        path = Path(v)
        parent = path.parent

        if not parent.exists():
            raise ValueError(f"Parent directory does not exist: {parent}")

        if not os.access(parent, os.W_OK):
            raise ValueError(f"Parent directory is not writable: {parent}")

        return v


# ── AUDIT LOGGER PROTOCOL ─────────────────────────────────────────────────────
class AuditLogger(Protocol):
    """Protocol defining the abstract audit logging interface.
    
    All audit consumers depend on this protocol, not the concrete implementation.
    Four methods: log, entries_by_task, entries_by_type, entries_in_range.
    No delete/update/clear methods — append-only by design.
    """
    
    def log(self, entry: AuditEntry) -> None:
        """Append a single AuditEntry to the audit log."""
        ...
    
    def entries_by_task(self, task_name: str) -> list[AuditEntry]:
        """Query the audit log for all entries matching the given task_name."""
        ...
    
    def entries_by_type(self, event_type: EventType) -> list[AuditEntry]:
        """Query the audit log for all entries matching the given event_type."""
        ...
    
    def entries_in_range(self, start: str, end: str) -> list[AuditEntry]:
        """Query the audit log for all entries with timestamps in [start, end]."""
        ...


# ── TYPE ALIAS ────────────────────────────────────────────────────────────────
AuditEntryList = list[AuditEntry]


# ── CUSTOM JSON FORMATTER ─────────────────────────────────────────────────────
class JsonFormatter(logging.Formatter):
    """Custom formatter that outputs log records as JSON lines."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON line (the record.msg is already JSON)."""
        # The message should already be JSON from our log() method
        return record.msg


# ── JSONLINES AUDIT LOGGER ────────────────────────────────────────────────────
class JsonLinesAuditLogger:
    """Concrete implementation of AuditLogger.
    
    Uses stdlib logging module with a custom JSON formatter attached to a
    dedicated 'apprentice.audit' named logger. File handler opened in append mode.
    Accepts AuditConfig via constructor (dependency injection).
    Thread-safe via stdlib logging internals.
    """
    
    def __init__(self, config: AuditConfig, event_dispatcher: Optional[Any] = None) -> None:
        """Construct a JsonLinesAuditLogger from an AuditConfig.

        Validates that log_path's parent directory exists and is writable.
        Creates the log file if it does not exist.
        Configures a dedicated stdlib logging.Logger named 'apprentice.audit'
        with a RotatingFileHandler using max_file_size_mb and backup_count.
        Honors log_level and log_to_stdout from config.
        Optionally accepts an EventDispatcher to route events to additional sinks.

        Args:
            config: Validated AuditConfig instance
            event_dispatcher: Optional EventDispatcher for multi-sink routing

        Raises:
            FileNotFoundError: Parent directory does not exist
            PermissionError: Parent directory is not writable
            ValueError: Invalid AuditConfig
        """
        if not isinstance(config, AuditConfig):
            raise ValueError(f"Invalid AuditConfig: {config}")

        self.config = config
        self._event_dispatcher = event_dispatcher
        log_path = Path(config.log_path)
        parent = log_path.parent

        # Validate parent directory exists
        if not parent.exists():
            raise FileNotFoundError(f"Parent directory does not exist: {parent}")

        # Validate parent directory is writable
        if not os.access(parent, os.W_OK):
            raise PermissionError(f"Cannot write to audit log path: {config.log_path}")

        # Create the log file if it doesn't exist
        if not log_path.exists():
            log_path.touch()

        # Check if the file itself is writable (if it already exists)
        if log_path.exists() and not os.access(log_path, os.W_OK):
            raise PermissionError(f"Cannot write to audit log path: {config.log_path}")

        # Resolve log level from config
        log_level = getattr(logging, config.log_level.upper(), logging.INFO)

        # Set up the dedicated logger
        self._logger = logging.getLogger('apprentice.audit')
        self._logger.setLevel(log_level)
        self._logger.propagate = False  # Isolate from root logger

        # Remove any existing handlers to avoid duplicates
        self._logger.handlers.clear()

        # Create RotatingFileHandler with config-driven rotation
        max_bytes = config.max_file_size_mb * 1024 * 1024
        file_handler = RotatingFileHandler(
            config.log_path,
            maxBytes=max_bytes,
            backupCount=config.backup_count,
            encoding='utf-8',
        )
        file_handler.setLevel(log_level)

        # Set custom JSON formatter
        json_formatter = JsonFormatter()
        file_handler.setFormatter(json_formatter)

        # Add handler to logger
        self._logger.addHandler(file_handler)

        # Add stdout handler if configured
        if config.log_to_stdout:
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setLevel(log_level)
            stdout_handler.setFormatter(json_formatter)
            self._logger.addHandler(stdout_handler)

        _log('info', f"Initialized audit logger with log_path={config.log_path}")
    
    def log(self, entry: AuditEntry) -> None:
        """Append a single AuditEntry to the audit log.

        Serializes the entry via to_json_line() and emits it through the
        stdlib 'apprentice.audit' logger, which writes to the configured JSONL
        file handler. Also dispatches to the EventDispatcher if configured.
        Synchronous — blocks until the entry is flushed.
        Thread-safe via stdlib logging internals.

        Args:
            entry: Valid AuditEntry instance

        Raises:
            OSError: Log file has become inaccessible
            ValueError: Entry's details dict contains non-JSON-serializable value
        """
        try:
            json_line = entry.to_json_line()
            self._logger.info(json_line)
            # Flush to ensure immediate write to disk
            for handler in self._logger.handlers:
                handler.flush()
            # Dispatch to additional sinks if configured
            if self._event_dispatcher is not None:
                self._event_dispatcher.emit(entry)
        except ValueError as e:
            raise ValueError(f"Cannot serialize audit entry: {e}")
        except OSError as e:
            raise OSError(f"Failed to write audit entry: {e}")
    
    def _read_entries(self) -> list[AuditEntry]:
        """Read all entries from the audit log file.
        
        Helper method that stream-reads the JSONL file line by line,
        deserializing each into an AuditEntry. Corrupted lines are skipped
        with a warning logged to the application logger.
        
        Returns:
            List of all valid AuditEntry objects in file order
            
        Raises:
            FileNotFoundError: Log file has been deleted
            PermissionError: Log file is not readable
        """
        log_path = Path(self.config.log_path)
        
        if not log_path.exists():
            raise FileNotFoundError(f"Audit log file not found: {self.config.log_path}")
        
        if not os.access(log_path, os.R_OK):
            raise PermissionError(f"Cannot read audit log file: {self.config.log_path}")
        
        entries = []
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    entry = AuditEntry(**data)
                    entries.append(entry)
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    _log('warning', f"Skipping corrupted line {line_num}: {e}")
                    continue
        
        return entries
    
    def entries_by_task(self, task_name: str) -> AuditEntryList:
        """Query the audit log for all entries matching the given task_name.
        
        Stream-reads the JSONL file line by line, deserializing each into an
        AuditEntry and filtering by task_name equality (case-sensitive exact match).
        Corrupted or unparseable lines are skipped with a warning.
        Returns entries in chronological order (file order).
        
        Args:
            task_name: Non-empty task name to filter by
            
        Returns:
            List of AuditEntry objects matching task_name, in chronological order
            
        Raises:
            FileNotFoundError: Log file has been deleted
            PermissionError: Log file is not readable
            ValueError: task_name is empty
        """
        if not task_name or len(task_name.strip()) == 0:
            raise ValueError("task_name must be non-empty")
        
        all_entries = self._read_entries()
        return [e for e in all_entries if e.task_name == task_name]
    
    def entries_by_type(self, event_type: EventType) -> AuditEntryList:
        """Query the audit log for all entries matching the given event_type.
        
        Stream-reads the JSONL file line by line, deserializing each into an
        AuditEntry and filtering by event_type equality.
        Corrupted or unparseable lines are skipped with a warning.
        Returns entries in chronological order (file order).
        
        Args:
            event_type: Valid EventType variant to filter by
            
        Returns:
            List of AuditEntry objects matching event_type, in chronological order
            
        Raises:
            FileNotFoundError: Log file has been deleted
            PermissionError: Log file is not readable
            ValueError: event_type is not a valid EventType variant
        """
        if not isinstance(event_type, EventType):
            raise ValueError(
                f"Invalid event type: {event_type}. "
                f"Must be one of the 9 defined EventType variants."
            )
        
        all_entries = self._read_entries()
        return [e for e in all_entries if e.event_type == event_type]
    
    def entries_in_range(self, start: str, end: str) -> AuditEntryList:
        """Query the audit log for all entries with timestamps in [start, end].
        
        Stream-reads the JSONL file line by line, deserializing each into an
        AuditEntry and filtering by timestamp comparison (inclusive on both bounds).
        Both start and end must be timezone-aware UTC datetimes in ISO 8601 format.
        Corrupted or unparseable lines are skipped with a warning.
        Returns entries in chronological order (file order).
        
        Args:
            start: ISO 8601 UTC datetime string (inclusive lower bound)
            end: ISO 8601 UTC datetime string (inclusive upper bound)
            
        Returns:
            List of AuditEntry objects within [start, end], in chronological order
            
        Raises:
            FileNotFoundError: Log file has been deleted
            PermissionError: Log file is not readable
            ValueError: start or end is invalid, or start > end
        """
        # Validate and parse start
        try:
            start_dt = datetime.fromisoformat(start)
            if start_dt.tzinfo is None:
                raise ValueError("start must be a timezone-aware UTC datetime in ISO 8601 format")
        except (ValueError, TypeError) as e:
            raise ValueError("start must be a timezone-aware UTC datetime in ISO 8601 format")
        
        # Validate and parse end
        try:
            end_dt = datetime.fromisoformat(end)
            if end_dt.tzinfo is None:
                raise ValueError("end must be a timezone-aware UTC datetime in ISO 8601 format")
        except (ValueError, TypeError) as e:
            raise ValueError("end must be a timezone-aware UTC datetime in ISO 8601 format")
        
        # Check start <= end
        if start_dt > end_dt:
            raise ValueError(f"start must be <= end; got start={start}, end={end}")
        
        all_entries = self._read_entries()
        result = []
        
        for entry in all_entries:
            try:
                entry_dt = datetime.fromisoformat(entry.timestamp)
                if start_dt <= entry_dt <= end_dt:
                    result.append(entry)
            except (ValueError, TypeError):
                # Skip entries with invalid timestamps
                continue
        
        return result


# ── REQUIRED EXPORTS ──────────────────────────────────────────────────────────
__all__ = [
    'EventType',
    'AuditEntry',
    'AuditConfig',
    'AuditLogger',
    'JsonLinesAuditLogger',
    'AuditEntryList',
]
