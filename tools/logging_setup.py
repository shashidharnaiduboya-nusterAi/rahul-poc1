"""
tools/logging_setup.py -- Python logging configuration
========================================================
Centralised logging setup for the POC-1 pipeline.

Design goals:
  - Console handler at INFO (human-friendly) and a rotating file handler
    at DEBUG (deep diagnostics) writing to ``data/logs/poc1.log``.
  - Every log record carries ``alert_id`` and ``step`` as structured fields so
    lines can be filtered per-alert with ``grep 'alert_id=<id>'``.
  - Any module can get a logger with :func:`get_logger` and optionally bind a
    per-alert context with :func:`bind_alert`.
  - Configuration is idempotent -- calling :func:`setup_logging` twice is safe.

Env vars:
  LOG_LEVEL_CONSOLE   -- console log level (default INFO)
  LOG_LEVEL_FILE      -- file log level (default DEBUG)
  LOG_DIR             -- directory for the rotating log file (default data/logs)
  LOG_FILE            -- filename (default poc1.log)
  LOG_MAX_BYTES       -- rotation size in bytes (default 10 MB)
  LOG_BACKUP_COUNT    -- number of rotated backups to keep (default 5)
"""

from __future__ import annotations

import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_INITIALISED = False

_BASE_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_LOG_DIR = _BASE_DIR / "data" / "logs"
_DEFAULT_LOG_FILE = "poc1.log"

_FMT = (
    "%(asctime)s %(levelname)-5s [%(name)s] "
    "alert_id=%(alert_id)s step=%(step)s %(message)s"
)


class _ContextFilter(logging.Filter):
    """Ensures every record has ``alert_id`` and ``step`` so the formatter never KeyErrors."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401 - short
        if not hasattr(record, "alert_id"):
            record.alert_id = "-"
        if not hasattr(record, "step"):
            record.step = "-"
        return True


def setup_logging(force: bool = False) -> None:
    """
    Configure root logger with console + rotating file handlers.

    Safe to call repeatedly; only sets up once unless ``force=True``.
    """
    global _INITIALISED
    if _INITIALISED and not force:
        return

    console_level = _level_from_env("LOG_LEVEL_CONSOLE", logging.INFO)
    file_level = _level_from_env("LOG_LEVEL_FILE", logging.DEBUG)

    log_dir = Path(os.getenv("LOG_DIR", str(_DEFAULT_LOG_DIR)))
    log_file = os.getenv("LOG_FILE", _DEFAULT_LOG_FILE)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        log_dir = _DEFAULT_LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))
    backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    if force:
        for h in list(root.handlers):
            root.removeHandler(h)

    ctx_filter = _ContextFilter()
    formatter = logging.Formatter(_FMT)

    stream = logging.StreamHandler()
    stream.setLevel(console_level)
    stream.setFormatter(formatter)
    stream.addFilter(ctx_filter)
    root.addHandler(stream)

    file_handler = RotatingFileHandler(
        str(log_dir / log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(ctx_filter)
    root.addHandler(file_handler)

    # Silence noisy third-party libs unless explicitly requested.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    _INITIALISED = True

    root.info(
        "logging initialised  console=%s file=%s path=%s",
        logging.getLevelName(console_level),
        logging.getLevelName(file_level),
        log_dir / log_file,
        extra={"alert_id": "-", "step": "logging_setup"},
    )


def _level_from_env(env_var: str, default: int) -> int:
    val = os.getenv(env_var, "")
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return getattr(logging, val.upper(), default)


def get_logger(name: str) -> logging.Logger:
    """Return a logger; initialises the root config on first call."""
    setup_logging()
    return logging.getLogger(name)


def bind_alert(
    logger: logging.Logger,
    alert_id: str,
    step: Optional[str] = None,
) -> logging.LoggerAdapter:
    """
    Return a :class:`LoggerAdapter` that injects ``alert_id`` / ``step`` on every
    record emitted through it.

    ``step`` can be overridden per-call via ``extra={"step": ...}``.
    """
    setup_logging()
    extra = {"alert_id": alert_id or "-", "step": step or "-"}
    return _BoundAdapter(logger, extra)


class _BoundAdapter(logging.LoggerAdapter):
    """LoggerAdapter that merges per-call ``extra`` on top of the bound context."""

    def process(self, msg, kwargs):
        extra = dict(self.extra)
        if "extra" in kwargs and isinstance(kwargs["extra"], dict):
            extra.update(kwargs["extra"])
        kwargs["extra"] = extra
        return msg, kwargs

    def with_step(self, step: str) -> "_BoundAdapter":
        """Return a new adapter with ``step`` rebound but same alert_id."""
        new_extra = dict(self.extra)
        new_extra["step"] = step
        return _BoundAdapter(self.logger, new_extra)


class StepTimer:
    """
    Context manager that logs the start/end of a step and its wall time.

    Usage::

        log = bind_alert(get_logger(__name__), "ABC123")
        with StepTimer(log, "case_lookup"):
            ...
    """

    def __init__(self, log: logging.Logger, step: str, level: int = logging.INFO):
        self._log = log
        self._step = step
        self._level = level
        self._t0 = 0.0

    def __enter__(self) -> "StepTimer":
        self._t0 = time.perf_counter()
        self._log.log(self._level, "START %s", self._step, extra={"step": self._step})
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        dt_ms = (time.perf_counter() - self._t0) * 1000.0
        if exc_type is None:
            self._log.log(
                self._level,
                "END   %s elapsed_ms=%.1f",
                self._step,
                dt_ms,
                extra={"step": self._step},
            )
        else:
            self._log.error(
                "FAIL  %s elapsed_ms=%.1f exc=%s: %s",
                self._step,
                dt_ms,
                exc_type.__name__,
                exc,
                extra={"step": self._step},
            )
        return False
