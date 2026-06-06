"""Prevent secrets (Telegram bot tokens, etc.) from appearing in deployment logs."""

from __future__ import annotations

import logging
import re

# Telegram Bot API URLs embed the full token: .../bot<token>/method
_BOT_URL_RE = re.compile(
    r"(https://api\.telegram\.org/bot)\d+:[A-Za-z0-9_-]+",
    re.IGNORECASE,
)
# Bare token pattern (id:secret)
_BARE_TOKEN_RE = re.compile(r"\b\d{8,}:[A-Za-z0-9_-]{20,}\b")


class _RedactSecretsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        redacted = _BARE_TOKEN_RE.sub("***TELEGRAM_TOKEN_REDACTED***", msg)
        redacted = _BOT_URL_RE.sub(r"\1***REDACTED***", redacted)
        if redacted != msg:
            record.msg = redacted
            record.args = ()
        return True


def configure_safe_logging(level: int = logging.INFO) -> None:
    """Call once at process startup (portal, tg-bot, workers)."""
    logging.basicConfig(level=level, force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    root = logging.getLogger()
    if not any(isinstance(f, _RedactSecretsFilter) for f in root.filters):
        root.addFilter(_RedactSecretsFilter())
