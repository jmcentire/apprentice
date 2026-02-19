"""Configurable event dispatch system for Apprentice.

Routes AuditEntry events to multiple sinks (file, stdout, socket, HTTP,
callable) using Python's stdlib logging handlers as the foundation.
"""

import importlib
import json
import logging
import sys
from logging.handlers import RotatingFileHandler, SocketHandler, HTTPHandler
from typing import Any, Callable, Optional

from pydantic import BaseModel


class _JsonEventFormatter(logging.Formatter):
    """Formats log records as compact JSON lines (the message is already JSON)."""

    def format(self, record: logging.LogRecord) -> str:
        return record.msg


class _TextEventFormatter(logging.Formatter):
    """Formats log records as human-readable text."""

    def format(self, record: logging.LogRecord) -> str:
        try:
            data = json.loads(record.msg)
            ts = data.get("timestamp", "")
            event = data.get("event_type", "")
            task = data.get("task_name", "")
            return f"[{ts}] {event} task={task}"
        except (json.JSONDecodeError, TypeError):
            return record.msg


class _CallableHandler(logging.Handler):
    """Handler that invokes a user-supplied callable for each event."""

    def __init__(self, callback: Callable[[str], None], level: int = logging.NOTSET):
        super().__init__(level)
        self._callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._callback(msg)
        except Exception:
            self.handleError(record)


def _resolve_callable(dotted_path: str) -> Callable:
    """Import a callable from a dotted module path like 'pkg.mod.func'."""
    module_path, _, attr_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid callable path: {dotted_path}")
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def _level_from_string(level_str: str) -> int:
    """Convert a log level string to its numeric value."""
    numeric = getattr(logging, level_str.upper(), None)
    if numeric is None:
        raise ValueError(f"Invalid log level: {level_str}")
    return numeric


class EventDispatcher:
    """Routes AuditEntry events to multiple configured handlers.

    Each handler config maps to a Python logging.Handler:
      "file"     -> RotatingFileHandler
      "stdout"   -> StreamHandler(sys.stdout)
      "socket"   -> SocketHandler(host, port)
      "http"     -> HTTPHandler(host, url, method)
      "callable" -> _CallableHandler wrapping an imported callable
    """

    def __init__(
        self,
        handler_configs: list[Any],
        base_logger_name: str = "apprentice.events",
    ):
        self._logger = logging.getLogger(base_logger_name)
        self._logger.setLevel(logging.DEBUG)  # Let individual handlers filter
        self._logger.propagate = False
        self._logger.handlers.clear()

        for cfg in handler_configs:
            handler = self._build_handler(cfg)
            if handler is not None:
                self._logger.addHandler(handler)

    def _build_handler(self, cfg: Any) -> Optional[logging.Handler]:
        """Build a logging.Handler from an EventHandlerConfig-like object."""
        handler_type = cfg.type if hasattr(cfg, "type") else cfg.get("type", "")
        level_str = cfg.level if hasattr(cfg, "level") else cfg.get("level", "INFO")
        fmt_str = getattr(cfg, "format", "json") if hasattr(cfg, "format") else cfg.get("format", "json")
        level = _level_from_string(level_str)

        formatter = _JsonEventFormatter() if fmt_str == "json" else _TextEventFormatter()

        handler: Optional[logging.Handler] = None

        if handler_type == "file":
            path = cfg.path if hasattr(cfg, "path") else cfg.get("path", "")
            max_bytes = int(getattr(cfg, "max_bytes", 0) or 0)
            backup_count = int(getattr(cfg, "backup_count", 0) or 0)
            if max_bytes > 0:
                handler = RotatingFileHandler(
                    path, maxBytes=max_bytes, backupCount=backup_count,
                    encoding="utf-8",
                )
            else:
                handler = logging.FileHandler(path, mode="a", encoding="utf-8")

        elif handler_type == "stdout":
            handler = logging.StreamHandler(sys.stdout)

        elif handler_type == "socket":
            host = cfg.host if hasattr(cfg, "host") else cfg.get("host", "localhost")
            port = int(cfg.port if hasattr(cfg, "port") else cfg.get("port", 0))
            handler = SocketHandler(host, port)

        elif handler_type == "http":
            host = cfg.host if hasattr(cfg, "host") else cfg.get("host", "")
            url = cfg.url if hasattr(cfg, "url") else cfg.get("url", "")
            method = cfg.method if hasattr(cfg, "method") else cfg.get("method", "POST")
            handler = HTTPHandler(host, url, method=method)

        elif handler_type == "callable":
            dotted = cfg.handler if hasattr(cfg, "handler") else cfg.get("handler", "")
            callback = _resolve_callable(dotted)
            handler = _CallableHandler(callback, level)

        if handler is not None:
            handler.setLevel(level)
            handler.setFormatter(formatter)

        return handler

    def emit(self, entry: Any) -> None:
        """Dispatch an event to all configured handlers.

        Accepts an AuditEntry (or anything with a to_json_line() method).
        """
        json_line = entry.to_json_line()
        self._logger.info(json_line)

    def close(self) -> None:
        """Flush and close all handlers."""
        for handler in self._logger.handlers[:]:
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)
