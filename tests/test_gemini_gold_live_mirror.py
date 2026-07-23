"""Gemini Gold live mirror config + guardrail tests."""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gemini_gold_trader.config import EXECUTION_MODE_DEMO, EXECUTION_MODE_LIVE
from app.gemini_gold_trader.models import GeminiGoldConfig
from app.gemini_gold_trader.routes import router


@pytest.fixture()
def api_client(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    GeminiGoldConfig.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)
    bootstrap = Session()
    bootstrap.add(
        GeminiGoldConfig(
            id=1,
            enabled=True,
            dry_run=False,
            demo_user_id=42,
            demo_ctrader_account_id="111",
            execution_mode=EXECUTION_MODE_DEMO,
            live_ctrader_account_id="222",
            live_mirror_enabled=False,
            max_live_trades_day=3,
        )
    )
    bootstrap.commit()
    bootstrap.close()

    admin = MagicMock(id=42, uid="TH-TEST", is_admin=True)

    monkeypatch.setattr("app.gemini_gold_trader.routes.SessionLocal", Session)
    monkeypatch.setattr(
        "app.gemini_gold_trader.routes.ensure_gemini_gold_trader_schema",
        lambda *a, **k: None,
    )
    monkeypatch.setattr("app.gemini_gold_trader.routes._resolve_user", lambda uid, db: admin)
    monkeypatch.setattr(
        "app.gemini_gold_trader.routes.live_accounts_for_user_id",
        lambda db, uid: [{"ctid": "222", "label": "#222 Live"}],
    )

    app = FastAPI()
    app.include_router(router)
    yield TestClient(app), Session, engine


def test_enable_live_mirror_persists(api_client):
    client, Session, engine = api_client
    r = client.post(
        "/api/gemini-gold-trader/config?uid=TH-TEST",
        json={
            "live_mirror_enabled": True,
            "live_ctrader_account_id": "222",
            "confirm_real_money": True,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["live_mirror_enabled"] is True
    assert body["config"]["live_mirror_enabled"] is True
    check = Session(bind=engine)
    row = check.query(GeminiGoldConfig).filter_by(id=1).first()
    assert row.live_mirror_enabled is True
    assert row.live_mirror_confirmed_at is not None
    check.close()


def test_enable_live_mirror_requires_confirm(api_client):
    client, _Session, _engine = api_client
    r = client.post(
        "/api/gemini-gold-trader/config?uid=TH-TEST",
        json={"live_mirror_enabled": True, "live_ctrader_account_id": "222"},
    )
    assert r.status_code == 400
    assert "confirm_real_money" in r.json()["detail"]


def test_enable_live_mirror_blocks_live_execution_mode(api_client):
    client, Session, engine = api_client
    check = Session(bind=engine)
    row = check.query(GeminiGoldConfig).filter_by(id=1).first()
    row.execution_mode = EXECUTION_MODE_LIVE
    check.commit()
    check.close()
    r = client.post(
        "/api/gemini-gold-trader/config?uid=TH-TEST",
        json={
            "live_mirror_enabled": True,
            "live_ctrader_account_id": "222",
            "confirm_real_money": True,
        },
    )
    assert r.status_code == 400
    assert "demo execution mode" in r.json()["detail"].lower()


def test_disconnect_live_mirror(api_client):
    client, Session, engine = api_client
    check = Session(bind=engine)
    row = check.query(GeminiGoldConfig).filter_by(id=1).first()
    row.live_mirror_enabled = True
    check.commit()
    r = client.post("/api/gemini-gold-trader/disconnect-live?uid=TH-TEST")
    assert r.status_code == 200
    assert r.json()["live_mirror_enabled"] is False
    check.refresh(row)
    assert row.live_mirror_enabled is False
    assert row.live_mirror_confirmed_at is None
    check.close()


def _demo_cfg():
    from app.gemini_gold_trader.config import env_defaults

    cfg = env_defaults()
    return cfg.__class__(
        **{
            **cfg.__dict__,
            "execution_mode": EXECUTION_MODE_DEMO,
            "live_mirror_enabled": True,
            "demo_user_id": 42,
            "live_ctrader_account_id": "222",
        }
    )


def test_live_mirror_sizes_by_lots_not_demo_wire_volume():
    """Regression: the live mirror must size by LOTS (volume_lots), never by the
    demo's raw broker wire volume. demo_ex.broker_volume_units is lots × lotSize
    on the demo spec; passing it as volume_units (contracts) blew the live order
    up ~2000x → NOT_ENOUGH_MONEY."""
    import asyncio
    from unittest.mock import AsyncMock, patch

    from app.gemini_gold_trader import executor

    db = MagicMock()
    user = MagicMock(id=42)
    prefs = MagicMock(ctrader_access_token="tok")
    demo_ex = MagicMock(
        direction="SHORT",
        sl_price=4055.0,
        tp_price=4043.0,
        broker_volume_units=100,  # demo wire volume (0.01 lot × lotSize 10000)
    )
    place = AsyncMock(
        return_value={
            "actual_fill": 4049.0,
            "position_id": "555",
            "order_id": "777",
            "volume": 100,
        }
    )

    cfg = _demo_cfg()
    cfg = cfg.__class__(**{**cfg.__dict__, "live_lot_size": 0.01, "demo_lot_size": 0.01})

    async def _run():
        async def _in_db(fn, *a, **k):
            # execute_live_mirror_take calls run_in_db_thread with, in order:
            # _resolve_live_mirror_trader, a nested _load_demo_ex closure, and
            # ensure_system_strategy. Route the two module-level callables and
            # default the nested demo-load closure to the mocked demo execution.
            if fn is executor._resolve_live_mirror_trader:
                return user, prefs
            if fn is executor.ensure_system_strategy:
                return 7
            return demo_ex

        with patch.object(
            executor, "run_in_db_thread", side_effect=_in_db
        ), patch.object(
            executor, "is_live_execution_mode", return_value=False
        ), patch.object(
            executor, "assert_live_account", return_value=None
        ), patch(
            "app.services.ctrader_client.place_market_order_resilient", new=place
        ), patch.object(
            executor, "db_commit", new=AsyncMock()
        ):
            await executor.execute_live_mirror_take(
                db=db,
                cfg=cfg,
                decision={
                    "direction": "SHORT",
                    "entry": 4049.0,
                    "stop_loss": 4055.0,
                    "take_profit": 4043.0,
                    "confidence": 80,
                },
                decision_id=1801,
                demo_execution_id=11,
            )
        place.assert_awaited_once()
        kwargs = place.await_args.kwargs
        assert kwargs.get("volume_lots") == 0.01
        assert "volume_units" not in kwargs

    asyncio.run(_run())


def test_live_mirror_skip_sends_telegram_reason():
    """A blocked live mirror (enabled) must surface the reason on Telegram."""
    import asyncio
    from unittest.mock import AsyncMock, patch

    from app.gemini_gold_trader import executor

    db = MagicMock()
    notify = AsyncMock()

    async def _run():
        with patch.object(
            executor, "is_live_execution_mode", return_value=False
        ), patch(
            "app.gemini_gold_trader.guardrails.check_can_execute_live_mirror",
            return_value=(False, "max_live_open_position"),
        ), patch.object(
            executor, "db_commit", new=AsyncMock()
        ), patch(
            "app.gemini_gold_trader.telegram_notify.notify_live_mirror_skipped",
            new=notify,
        ), patch.object(
            executor, "execute_live_mirror_take", new=AsyncMock(return_value=None)
        ):
            await executor.maybe_live_mirror_after_demo(
                db=db,
                cfg=_demo_cfg(),
                decision={"direction": "LONG", "confidence": 80},
                decision_id=99,
                demo_execution_id=10,
                session="new_york",
            )
        notify.assert_awaited_once()
        kwargs = notify.await_args.kwargs
        assert kwargs["reason"] == "max_live_open_position"
        assert kwargs["status"] == "skipped"

    asyncio.run(_run())
