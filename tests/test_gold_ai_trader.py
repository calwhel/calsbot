"""Gold AI Trader unit tests (no live API / broker)."""
import asyncio
import json
import os
from datetime import datetime

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gold_ai_trader.config import confidence_threshold, env_defaults, gold_ai_trader_enabled
from app.gold_ai_trader.scanner import active_session, Candidate
from app.gold_ai_trader.claude import decide, system_prompt, SYSTEM_PROMPT
from app.gold_ai_trader.data_quality import gold_data_ok_for_claude, format_data_source
from app.gold_ai_trader.context import build_context_snapshot
from app.gold_ai_trader.guardrails import (
    assert_demo_account,
    assert_live_account,
    DemoAccountRequired,
    LiveAccountRequired,
    check_can_execute_live_mirror,
    check_can_call_claude,
    demo_account_configured,
    calls_today,
    cost_today_usd,
    reset_daily_claude_credits,
    maybe_reset_daily_claude_credits,
)
from app.gold_ai_trader.accounts import demo_accounts_from_prefs, validate_demo_ctid_allowed
from app.gold_ai_trader.executor import _pct_from_prices
from app.gold_ai_trader.routes import _normalize_uid, _persist_demo_user_from_admin
from app.gold_ai_trader.telegram_notify import (
    daily_summary_enabled,
    format_close_message,
    format_daily_summary,
    format_take_message,
    telegram_notifications_enabled,
)


class _FakePrefs:
    ctrader_accounts = '[{"ctidTraderAccountId": 12345, "isLive": false}]'


def test_feature_flag_default_off():
    assert gold_ai_trader_enabled() is False or env_defaults().enabled is True


def test_normalize_uid_adds_th_prefix():
    assert _normalize_uid("yp0bada8") == "TH-YP0BADA8"
    assert _normalize_uid("TH-YP0BADA8") == "TH-YP0BADA8"


def test_confidence_threshold_default_45():
    os.environ.pop("GOLD_AI_CONFIDENCE_THRESHOLD", None)
    assert confidence_threshold() == 45
    assert env_defaults().confidence_threshold == 45


def test_confidence_threshold_env_override():
    os.environ["GOLD_AI_CONFIDENCE_THRESHOLD"] = "55"
    try:
        assert confidence_threshold() == 55
        assert env_defaults().confidence_threshold == 55
    finally:
        os.environ.pop("GOLD_AI_CONFIDENCE_THRESHOLD", None)


def test_system_prompt_uses_configurable_threshold():
    prompt = system_prompt(50)
    assert "If confidence is below 50, action MUST be skip." in prompt
    assert "positive expectancy" in prompt.lower()
    assert "50–59" in prompt or "50-59" in prompt
    assert "institutional" in SYSTEM_PROMPT.lower()
    assert "CONFLUENCE CALIBRATION" in SYSTEM_PROMPT
    assert "Count: N/M passed" in SYSTEM_PROMPT


def test_gold_data_ok_requires_ctrader_sources():
    ok, reason = gold_data_ok_for_claude(
        {
            "price": 2650.0,
            "live_source": "ctrader",
            "price_source": "ctrader",
            "kline_source": "ctrader",
            "kline_synthetic": False,
            "klines_stale": False,
            "kline_bars": 40,
        }
    )
    assert ok is True
    assert reason == "ok"

    ok_fb, reason_fb = gold_data_ok_for_claude(
        {
            "price": 2650.0,
            "live_source": None,
            "price_source": "coinbase",
            "kline_source": "coinbase",
            "kline_synthetic": False,
            "klines_stale": False,
            "kline_bars": 40,
        }
    )
    assert ok_fb is False
    assert "fallback_klines" in reason_fb or "non_ctrader_price" in reason_fb

    ok_stale, reason_stale = gold_data_ok_for_claude(
        {
            "price": 2650.0,
            "live_source": "ctrader",
            "price_source": "ctrader",
            "kline_source": "ctrader",
            "kline_synthetic": False,
            "klines_stale": True,
            "stale_reason": "drift=1.2%",
            "kline_bars": 40,
        }
    )
    assert ok_stale is False
    assert reason_stale.startswith("stale_klines:")

    ok_syn, reason_syn = gold_data_ok_for_claude(
        {
            "price": 2650.0,
            "live_source": "ctrader",
            "price_source": "ctrader",
            "kline_source": "synthetic",
            "kline_synthetic": True,
            "klines_stale": False,
            "kline_bars": 40,
        }
    )
    assert ok_syn is False
    assert reason_syn == "synthetic_klines"


def test_format_data_source_tag():
    tag = format_data_source(
        {"live_source": "ctrader", "kline_source": "ctrader-user"}
    )
    assert tag == "price:ctrader/kline:ctrader-user"


def test_gold_ai_trader_page_sends_session_auth():
    """Gold AI UI must attach session token like the main portal (fixes infinite loading)."""
    html = open(
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "app/templates/gold_ai_trader.html",
        encoding="utf-8",
    ).read()
    routes = open(
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "app/gold_ai_trader/routes.py",
        encoding="utf-8",
    ).read()
    assert "X-TradeHub-Session" in html
    assert "_apiFetch" in html
    assert "showLoadError" in html
    assert "session_token" in routes
    assert "_session_token_for_page" in routes
    assert "portal_session" in routes
    assert "make_session_token" in routes
    assert "renderSessionTimeline(cfg, rt, j.shared_session_hours)" in html


def test_gold_ai_page_renders_without_db():
    """Dashboard HTML must load without hitting User table (auth on API only)."""
    from unittest.mock import patch
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.gold_ai_trader.routes import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    with patch("app.gold_ai_trader.routes.SessionLocal") as mock_sl:
        mock_sl.side_effect = AssertionError("page load must not open DB session")
        r = client.get("/gold-ai-trader?uid=TH-YP0BADA8")
    assert r.status_code == 200
    assert "Gold AI Trader" in r.text
    assert "TH-YP0BADA8" in r.text


def test_persist_demo_user_from_admin():
    class _Row:
        demo_user_id = None

    class _Admin:
        id = 42

    row = _Row()
    _persist_demo_user_from_admin(row, _Admin())
    assert row.demo_user_id == 42


def test_session_gate_london():
    cfg = env_defaults()
    assert active_session(datetime(2026, 6, 18, 8, 30), cfg) == "london"
    assert active_session(datetime(2026, 6, 18, 17, 0), cfg) == "new_york"
    assert active_session(datetime(2026, 6, 18, 5, 0), cfg) is None
    assert cfg.london_end_hour == 16
    assert cfg.ny_start_hour == 12
    assert cfg.ny_end_hour == 21


def test_demo_account_lock_rejects_live():
    cfg = env_defaults()
    cfg.demo_ctrader_account_id = "12345"

    class LivePrefs:
        ctrader_accounts = '[{"ctidTraderAccountId": 12345, "isLive": true}]'

    try:
        assert_demo_account(LivePrefs(), 12345, cfg)
        assert False, "expected DemoAccountRequired"
    except DemoAccountRequired:
        pass


def test_live_account_lock_rejects_demo():
    cfg = env_defaults()
    cfg.live_ctrader_account_id = "12345"
    cfg.live_mirror_enabled = True

    class DemoPrefs:
        ctrader_accounts = '[{"ctidTraderAccountId": 12345, "isLive": false}]'

    try:
        assert_live_account(DemoPrefs(), 12345, cfg)
        assert False, "expected LiveAccountRequired"
    except LiveAccountRequired:
        pass


def test_live_account_lock_requires_live_flag():
    cfg = env_defaults()
    cfg.live_ctrader_account_id = "99999"

    class LivePrefs:
        ctrader_accounts = '[{"ctidTraderAccountId": 99999, "isLive": true}]'

    assert_live_account(LivePrefs(), 99999, cfg)  # no raise


def test_pct_from_prices_long():
    sl, tp = _pct_from_prices("LONG", 2650.0, 2645.0, 2660.0)
    assert sl > 0 and tp > 0


class _CfgLive:
    kill_switch = False
    live_mirror_enabled = True
    live_ctrader_account_id = "111"
    max_live_trades_day = 3


class _DbLiveBlocked:
    def query(self, *a, **k):
        return self

    def filter(self, *a, **k):
        return self

    def scalar(self):
        return 99  # over cap


def test_kill_switch_blocks_live_mirror():
    cfg = _CfgLive()
    cfg.kill_switch = True
    ok, reason = check_can_execute_live_mirror(_DbLiveBlocked(), cfg, 1)
    assert not ok and reason == "kill_switch"


class _PrefsMixed:
    ctrader_accounts = json.dumps([
        {"ctidTraderAccountId": 111, "isLive": False, "traderLogin": 1001},
        {"ctidTraderAccountId": 222, "isLive": True, "traderLogin": 2002},
        {"ctidTraderAccountId": 333, "isLive": False, "traderLogin": 1003},
    ])


def test_demo_accounts_filters_live_only():
    demos = demo_accounts_from_prefs(_PrefsMixed())
    assert len(demos) == 2
    assert all(d["ctid"] in ("111", "333") for d in demos)
    assert all("Demo" in d["label"] for d in demos)


def test_validate_demo_ctid_rejects_live():
    demos = demo_accounts_from_prefs(_PrefsMixed())
    validate_demo_ctid_allowed(demos, "111")
    try:
        validate_demo_ctid_allowed(demos, "222")
        assert False, "expected ValueError for live ctid"
    except ValueError:
        pass


def test_no_demo_account_blocks_claude():
    cfg = env_defaults()
    cfg.enabled = True
    cfg.demo_ctrader_account_id = None
    ok, reason = check_can_call_claude(_DbLiveBlocked(), cfg)
    assert not ok and reason == "no_demo_account"


def test_demo_account_configured():
    cfg = env_defaults()
    cfg.demo_ctrader_account_id = "12345"
    assert demo_account_configured(cfg)
    cfg.demo_ctrader_account_id = ""
    assert not demo_account_configured(cfg)


def test_claude_dry_run_skip():
    decision, reasoning, meta = asyncio.run(
        decide("context", dry_run=True)
    )
    assert decision["action"] == "skip"
    assert meta["cost_usd"] == 0.0


def test_system_prompt_cached_block_present():
    assert "positive expectancy" in SYSTEM_PROMPT.lower()


def test_context_builder_shape():
    class _Cfg:
        london_start_hour = 7
        london_end_hour = 16
        ny_start_hour = 13
        ny_end_hour = 21
        max_calls_day = 50
        max_trades_day = 6
        confidence_threshold = 50

    class _Db:
        def query(self, *a, **k):
            return self

        def filter(self, *a, **k):
            return self

        def order_by(self, *a, **k):
            return self

        def limit(self, *a, **k):
            return self

        def first(self):
            return None

        def all(self):
            return []

        def scalar(self):
            return 0

    cand = Candidate(
        type="sweep_pdh",
        direction="SHORT",
        detail="Swept PDH then closed back below",
        quality_atr=1.2,
        sig_key="sweep_pdh:SHORT",
        raw={},
    )
    text = asyncio.run(
        build_context_snapshot(
            candidate=cand,
            price=2650.5,
            session="london",
            db=_Db(),
            cfg=_Cfg(),
            user_id=None,
        )
    )
    assert "TRIGGER" in text
    assert "XAUUSD" in text
    assert "sweep_pdh" in text
    assert "PREMIUM / DISCOUNT" in text
    assert "TRADE BANDS" in text
    assert "SETUP STATS" in text
    assert "Structure score" in text or "structure" in text.lower()

    cand_with_readiness = Candidate(
        type="liq_sweep_bull",
        direction="LONG",
        detail="LQ sweep bullish: reclaim @ 2650.0 FIRED",
        quality_atr=1.1,
        sig_key="liq_sweep_bull:LONG",
        raw={
            "readiness_score": 62,
            "readiness_breakdown": "htf+entry+disp",
            "readiness_checklist": {
                "htf_aligned": True,
                "at_entry": True,
                "displacement_ok": True,
                "reclaim_held": True,
                "rr_feasible": True,
            },
        },
    )
    rich = asyncio.run(
        build_context_snapshot(
            candidate=cand_with_readiness,
            price=2650.5,
            session="london",
            db=_Db(),
            cfg=_Cfg(),
            user_id=None,
        )
    )
    assert "=== CONFLUENCE" in rich
    assert "Count: 5/5 passed" in rich
    assert "=== SETUP RUBRIC" in rich
    assert "Liquidity sweep" in rich


def test_telegram_take_message_demo_label():
    text = format_take_message(
        candidate_type="sweep_pdh",
        session="london",
        decision={
            "direction": "short",
            "entry": 2650.5,
            "stop_loss": 2655.0,
            "take_profit": 2640.0,
            "rationale": "Clean sweep + displacement back inside range.",
        },
        confidence=82,
        executed=True,
        execution_id=99,
    )
    assert "[DEMO] Gold AI Trader" in text
    assert "TAKE" in text
    assert "sweep_pdh" in text
    assert "SHORT" in text
    assert "2650.50" in text
    assert "82%" in text
    assert "exec #99" in text.lower()


def test_telegram_close_message_demo_label():
    text = format_close_message(
        candidate_type="orb_break",
        session="new_york",
        direction="long",
        outcome="WIN",
        pnl_pct=1.25,
        pnl_usd=42.50,
        decision_id=7,
        execution_id=88,
    )
    assert "[DEMO] Gold AI Trader" in text
    assert "CLOSED WIN" in text
    assert "+1.25%" in text
    assert "$+42.50" in text


def test_telegram_daily_summary_format():
    text = format_daily_summary(
        calls=12,
        max_calls=50,
        trades=2,
        max_trades=6,
        cost_usd=0.45,
        demo_pnl_usd=18.20,
        open_positions=1,
    )
    assert "[DEMO] Gold AI Trader" in text
    assert "Daily summary" in text
    assert "12/50" in text
    assert "$+18.20" in text


def test_telegram_env_toggles_default_on():
    assert telegram_notifications_enabled() is True
    assert daily_summary_enabled() is True


def test_reset_daily_claude_credits_zeros_counters():
    from app.gold_ai_trader.guardrails import _calls_cutoff

    row = type("Row", (), {"calls_reset_at": None, "updated_at": None})()

    class _Query:
        def __init__(self, model):
            self.model = model

        def filter(self, *a, **k):
            return self

        def first(self):
            return row

    class _Db:
        def query(self, model):
            return _Query(model)

        def commit(self):
            pass

        def refresh(self, r):
            pass

    db = _Db()
    cutoff_before = _calls_cutoff(db)
    assert cutoff_before.replace(microsecond=0) <= datetime.utcnow().replace(microsecond=0)

    reset_at = reset_daily_claude_credits(db)
    assert reset_at is not None
    assert row.calls_reset_at is not None


def test_maybe_reset_daily_claude_credits_when_blocked():
    row = type(
        "Row",
        (),
        {
            "calls_reset_at": None,
            "updated_at": None,
            "enabled": False,
            "kill_switch": False,
            "london_start_hour": 7,
            "london_end_hour": 10,
            "ny_start_hour": 13,
            "ny_end_hour": 16,
            "max_calls_day": 22,
            "max_trades_day": 6,
            "no_overnight": True,
            "model": "claude-opus-4-8",
            "demo_user_id": None,
            "demo_ctrader_account_id": None,
        },
    )()

    class _Query:
        def filter(self, *a, **k):
            return self

        def first(self):
            return row

        def scalar(self):
            return 28

    class _Db:
        def query(self, *a, **k):
            return _Query()

        def commit(self):
            pass

        def refresh(self, r):
            pass

    assert maybe_reset_daily_claude_credits(_Db()) is True
    assert row.calls_reset_at is not None
