"""Gold AI status API + schema repair tests."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")


def test_apply_alters_uses_isolated_transactions():
    """Each ALTER must run in its own txn — PG aborts the batch on first error."""
    from app.gold_ai_trader import schema as schema_mod

    begin_calls = 0
    execute_calls = 0

    class _Conn:
        dialect = type("D", (), {"name": "postgresql"})()

        def execute(self, _sql):
            nonlocal execute_calls
            execute_calls += 1
            if execute_calls == 1:
                raise RuntimeError("simulated first alter failure")

    class _Begin:
        def __enter__(self):
            return _Conn()

        def __exit__(self, *_args):
            return False

    def _fake_begin():
        nonlocal begin_calls
        begin_calls += 1
        return _Begin()

    fake_insp = MagicMock()
    fake_insp.has_table.return_value = True
    fake_insp.get_columns.return_value = []

    with patch.object(schema_mod.engine, "begin", side_effect=_fake_begin), patch(
        "app.gold_ai_trader.schema.inspect", return_value=fake_insp
    ):
        schema_mod._apply_alters()

    total_alters = len(schema_mod._GOLD_AI_COLUMN_ALTERS)
    assert begin_calls == total_alters
    assert execute_calls == total_alters


def test_status_api_returns_ok_when_config_orm_fails():
    """Dashboard should load with env defaults when config row cannot be read."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.gold_ai_trader.routes import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    admin = MagicMock()
    admin.id = 42
    admin.uid = "TH-YP0BADA8"
    admin.is_admin = True

    mock_db = MagicMock()
    mock_db.rollback = MagicMock()
    mock_db.close = MagicMock()

    with patch("app.gold_ai_trader.routes.ensure_gold_ai_trader_schema"), patch(
        "app.gold_ai_trader.routes.SessionLocal", return_value=mock_db
    ), patch("app.gold_ai_trader.routes._resolve_user", return_value=admin), patch(
        "app.gold_ai_trader.routes.seed_config_if_missing",
        side_effect=RuntimeError("missing column live_mirror_enabled"),
    ), patch(
        "app.gold_ai_trader.routes.demo_accounts_for_user_id", return_value=[]
    ), patch("app.gold_ai_trader.routes._safe_lessons", return_value=[]), patch(
        "app.gold_ai_trader.routes._safe_decisions", return_value=[]
    ):
        r = client.get("/api/gold-ai-trader/status?uid=TH-YP0BADA8")

    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "config" in body
    assert body.get("degraded") and "config" in body["degraded"]


def test_status_api_offline_when_db_down_with_valid_session():
    """When Neon is unreachable, signed session still loads dashboard shell."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.gold_ai_trader.routes import router
    from app.portal_session import make_session_token
    from sqlalchemy.exc import OperationalError

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    token = make_session_token("TH-YP0BADA8")

    mock_db = MagicMock()
    mock_db.rollback = MagicMock()
    mock_db.close = MagicMock()
    mock_db.query.side_effect = OperationalError(
        "connect",
        {},
        Exception('endpoint has been disabled'),
    )

    with patch("app.gold_ai_trader.routes.ensure_gold_ai_trader_schema"), patch(
        "app.gold_ai_trader.routes.SessionLocal", return_value=mock_db
    ), patch("app.gold_ai_trader.routes._safe_lessons", return_value=[]), patch(
        "app.gold_ai_trader.routes._safe_decisions", return_value=[]
    ), patch("app.gold_ai_trader.routes._safe_funnel_events", return_value=[]):
        r = client.get(
            "/api/gold-ai-trader/status?uid=TH-YP0BADA8",
            headers={"X-TradeHub-Session": token},
        )

    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body.get("db_unavailable") is True
    assert "database" in body.get("degraded", [])


def test_is_db_connection_error_detects_neon_disabled():
    from sqlalchemy.exc import OperationalError
    from app.db_resilience import is_db_connection_error

    exc = OperationalError(
        "statement",
        {},
        Exception("The endpoint has been disabled. Enable it using the API and retry."),
    )
    assert is_db_connection_error(exc) is True


def test_schema_repair_detects_missing_columns():
    from sqlalchemy import create_engine, text

    from app.gold_ai_trader import schema as schema_mod

    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE gold_ai_config (
                    id INTEGER PRIMARY KEY,
                    enabled BOOLEAN NOT NULL DEFAULT 0,
                    kill_switch BOOLEAN NOT NULL DEFAULT 0,
                    london_start_hour INTEGER NOT NULL DEFAULT 7,
                    london_end_hour INTEGER NOT NULL DEFAULT 16,
                    ny_start_hour INTEGER NOT NULL DEFAULT 12,
                    ny_end_hour INTEGER NOT NULL DEFAULT 21,
                    max_calls_day INTEGER NOT NULL DEFAULT 22,
                    max_trades_day INTEGER NOT NULL DEFAULT 6,
                    no_overnight BOOLEAN NOT NULL DEFAULT 1,
                    model VARCHAR(64) NOT NULL DEFAULT 'claude-opus-4-8',
                    demo_ctrader_account_id VARCHAR(40),
                    demo_user_id INTEGER,
                    updated_at TIMESTAMP
                )
                """
            )
        )

    with patch.object(schema_mod, "engine", engine):
        missing = schema_mod._missing_columns()
        assert "gold_ai_config" in missing
        assert "live_mirror_enabled" in missing["gold_ai_config"]
        schema_mod._apply_alters()
        missing_after = schema_mod._missing_columns()
        assert "gold_ai_config" not in missing_after
