"""Structured logging with automatic secret redaction for ``--verbose`` and ``--log-file``."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{8,}|AIza[A-Za-z0-9_-]{12,}|gsk_[A-Za-z0-9_-]{10,}|"
    r"Bearer\s+[A-Za-z0-9._-]{8,}|x-api-key['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9._-]{6,})",
    re.I,
)


def redact_secrets(text: str) -> str:
    return _SECRET_RE.sub("[REDACTED]", text)


class _RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_secrets(record.msg)
        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(redact_secrets(a) if isinstance(a, str) else a for a in record.args)
            elif isinstance(record.args, dict):
                record.args = {k: redact_secrets(v) if isinstance(v, str) else v for k, v in record.args.items()}
        return True


class _JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_secrets(record.getMessage()),
        }
        if record.exc_info:
            payload["exc"] = redact_secrets(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=True)


def configure_logging(*, verbose: bool = False, log_file: Path | None = None) -> logging.Logger:
    """Configure the ``pullreel`` logger hierarchy for optional verbose stderr + JSONL file output."""
    logger = logging.getLogger("pullreel")
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    logger.propagate = False
    level = logging.DEBUG if (verbose or log_file is not None) else logging.WARNING
    logger.setLevel(level)

    filt = _RedactingFilter()
    if verbose:
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        sh.addFilter(filt)
        logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(_JsonLineFormatter())
        fh.addFilter(filt)
        logger.addHandler(fh)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger
