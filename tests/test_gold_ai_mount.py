"""Tests for Gold AI portal mount resilience."""
from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from unittest.mock import patch

from fastapi import FastAPI


def test_mount_registers_routes_even_if_schema_fails():
    from app.gold_ai_trader.portal_mount import mount_gold_ai_trader_portal

    app = FastAPI()
    with patch(
        "app.gold_ai_trader.schema.ensure_gold_ai_trader_schema",
        side_effect=RuntimeError("db unavailable"),
    ):
        mount_gold_ai_trader_portal(app)
    paths = [getattr(r, "path", None) for r in app.routes]
    assert "/gold-ai-trader" in paths


def test_mount_raises_when_schema_hard_fail_enabled(monkeypatch):
    from app.gold_ai_trader.portal_mount import mount_gold_ai_trader_portal

    app = FastAPI()
    monkeypatch.setenv("GOLD_AI_SCHEMA_STARTUP_HARD_FAIL", "true")
    with patch(
        "app.gold_ai_trader.schema.ensure_gold_ai_trader_schema",
        side_effect=RuntimeError("db unavailable"),
    ):
        raised = False
        try:
            mount_gold_ai_trader_portal(app)
        except RuntimeError:
            raised = True
        assert raised is True
