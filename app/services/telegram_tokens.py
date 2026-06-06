"""Resolve Telegram bot tokens consistently (settings + os.environ)."""

from __future__ import annotations

import os


def main_bot_token() -> str:
    """Active main bot token — pydantic settings first, then raw env."""
    try:
        from app.config import settings

        raw = getattr(settings, "TELEGRAM_BOT_TOKEN", None) or os.getenv("TELEGRAM_BOT_TOKEN")
    except Exception:
        raw = os.getenv("TELEGRAM_BOT_TOKEN")
    return (raw or "").strip()


def forex_bot_token() -> str:
    return (os.getenv("FOREX_BOT_TOKEN") or "").strip()


def tokens_are_same(a: str, b: str) -> bool:
    return bool(a and b and a.strip() == b.strip())


def should_run_forex_poller() -> bool:
    """True only when a distinct forex bot token is configured."""
    fx = forex_bot_token()
    if not fx:
        return False
    return not tokens_are_same(fx, main_bot_token())
