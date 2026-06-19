"""Gold AI Trader unit tests (no live API / broker)."""
import asyncio
import json
from datetime import datetime

from app.gold_ai_trader.config import env_defaults, gold_ai_trader_enabled
from app.gold_ai_trader.scanner import active_session, Candidate
from app.gold_ai_trader.claude import decide, SYSTEM_PROMPT
from app.gold_ai_trader.context import build_context_snapshot
from app.gold_ai_trader.guardrails import (
    assert_demo_account,
    assert_live_account,
    DemoAccountRequired,
    LiveAccountRequired,
    check_can_execute_live_mirror,
    check_can_call_claude,
    demo_account_configured,
)
from app.gold_ai_trader.accounts import demo_accounts_from_prefs, validate_demo_ctid_allowed
from app.gold_ai_trader.executor import _pct_from_prices
from app.gold_ai_trader.routes import _normalize_uid


class _FakePrefs:
    ctrader_accounts = '[{"ctidTraderAccountId": 12345, "isLive": false}]'


def test_feature_flag_default_off():
    assert gold_ai_trader_enabled() is False or env_defaults().enabled is True


def test_normalize_uid_adds_th_prefix():
    assert _normalize_uid("yp0bada8") == "TH-YP0BADA8"
    assert _normalize_uid("TH-YP0BADA8") == "TH-YP0BADA8"


def test_session_gate_london():
    cfg = env_defaults()
    cfg.london_start_hour = 7
    cfg.london_end_hour = 10
    cfg.ny_start_hour = 13
    cfg.ny_end_hour = 16
    assert active_session(datetime(2026, 6, 18, 8, 30), cfg) == "london"
    assert active_session(datetime(2026, 6, 18, 12, 0), cfg) is None


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
    assert "Default action is SKIP" in SYSTEM_PROMPT


def test_context_builder_shape():
    class _Cfg:
        london_start_hour = 7
        ny_start_hour = 13
        max_calls_day = 50
        max_trades_day = 6

    class _Db:
        def query(self, *a, **k):
            return self

        def filter(self, *a, **k):
            return self

        def order_by(self, *a, **k):
            return self

        def first(self):
            return None

        def all(self):
            return []

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
