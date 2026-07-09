"""Execution diagnostics and readiness checks."""
import os
from datetime import datetime, timedelta

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.gemini_gold_trader.config import env_defaults
from app.gemini_gold_trader.execution_diagnostics import (
    build_execution_readiness,
    skip_reason_breakdown,
    _normalize_skip_reason,
)
from app.gemini_gold_trader.guardrails import merge_config
from app.gemini_gold_trader.models import GeminiGoldConfig, GeminiGoldDecision


def test_normalize_skip_reason_strips_blocked_prefix():
    assert _normalize_skip_reason("blocked: dry_run") == "dry_run"
    assert _normalize_skip_reason("blocked: max_open_position — detail") == "max_open_position"


def test_skip_reason_breakdown_groups_blocked_takes():
    engine = create_engine("sqlite:///:memory:")
    GeminiGoldConfig.__table__.create(bind=engine)
    GeminiGoldDecision.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    now = datetime.utcnow()
    db.add(
        GeminiGoldDecision(
            ts=now,
            action="TAKE",
            executed=False,
            skip_reason="blocked: dry_run",
            confidence=85,
        )
    )
    db.add(
        GeminiGoldDecision(
            ts=now - timedelta(hours=1),
            action="TAKE",
            executed=False,
            skip_reason="blocked: dry_run",
            confidence=82,
        )
    )
    db.commit()
    rows = skip_reason_breakdown(db, days=14)
    assert rows[0]["reason"] == "dry_run"
    assert rows[0]["count"] == 2
    db.close()


def test_merge_config_env_dry_run_overrides_db():
    row = GeminiGoldConfig(id=1, dry_run=True, max_calls_day=340)
    env = env_defaults()
    env.dry_run = True
    os.environ["GEMINI_GOLD_DRY_RUN"] = "false"
    try:
        cfg = merge_config(row, env)
        assert cfg.dry_run is False
    finally:
        os.environ.pop("GEMINI_GOLD_DRY_RUN", None)


def test_build_execution_readiness_flags_dry_run():
    engine = create_engine("sqlite:///:memory:")
    GeminiGoldConfig.__table__.create(bind=engine)
    GeminiGoldDecision.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    row = GeminiGoldConfig(
        id=1,
        dry_run=True,
        demo_ctrader_account_id="47782488",
        demo_user_id=7,
        max_calls_day=340,
    )
    db.add(row)
    db.commit()
    cfg = merge_config(row, env_defaults())
    readiness = build_execution_readiness(
        db,
        cfg=cfg,
        user_id=7,
        account_snap={"ctrader_account_id": "47782488", "broker_unreachable": True},
    )
    assert readiness["ready"] is False
    assert any("Dry-run" in issue for issue in readiness["issues"])
    db.close()
