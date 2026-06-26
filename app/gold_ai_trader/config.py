"""Environment + runtime configuration for Gold AI Trader."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def gold_ai_trader_enabled() -> bool:
    return _env_bool("GOLD_AI_TRADER_ENABLED", False)


@dataclass
class GoldAiRuntimeConfig:
    enabled: bool
    kill_switch: bool
    london_start_hour: int
    london_end_hour: int
    ny_start_hour: int
    ny_end_hour: int
    max_calls_day: int
    max_trades_day: int
    no_overnight: bool
    scan_interval_s: float
    model: str
    demo_user_id: Optional[int]
    demo_ctrader_account_id: Optional[str]
    live_mirror_enabled: bool
    live_ctrader_account_id: Optional[str]
    live_lot_size: float
    demo_lot_size: float
    max_live_trades_day: int
    learning_every_n_closes: int
    min_lot: float
    use_limit_entry: bool
    pending_entry_timeout_min: int
    learning_daily_at_ny_end: bool
    confidence_threshold: int
    include_history_in_decisions: bool
    orb_enabled: bool
    orb_range_minutes: int
    orb_trade_window_minutes: int
    orb_timeframe: str
    orb_confirmation: str
    orb_require_retest: bool
    orb_retest_max_bars: int
    orb_retest_tol_atr: float
    orb_fakeout_filter: bool
    orb_break_buffer_atr: float
    orb_break_buffer_range_pct: float
    orb_min_break_body_atr: float
    orb_min_range_atr: float
    orb_max_range_atr: float
    orb_sl_mode: str
    orb_sl_atr_mult: float
    orb_sl_range_buffer_atr: float
    orb_tp_mode: str
    orb_tp_range_mult: float
    orb_tp_rr: float
    orb_confidence_threshold: int
    orb_max_calls_day: int
    orb_min_global_calls_left: int
    orb_max_trades_per_session: int
    orb_entry_max_break_atr: float
    orb_entry_max_break_range_pct: float


from app.services.forex_sessions import gold_ai_session_hours

_SHARED_SESSION_HOURS = gold_ai_session_hours()

DEFAULTS = GoldAiRuntimeConfig(
    enabled=False,
    kill_switch=False,
    london_start_hour=_SHARED_SESSION_HOURS["london"]["start_hour"],
    london_end_hour=_SHARED_SESSION_HOURS["london"]["end_hour"],
    ny_start_hour=_SHARED_SESSION_HOURS["new_york"]["start_hour"],
    ny_end_hour=_SHARED_SESSION_HOURS["new_york"]["end_hour"],
    max_calls_day=70,
    max_trades_day=6,
    no_overnight=True,
    scan_interval_s=20.0,
    model="claude-opus-4-8",
    demo_user_id=None,
    demo_ctrader_account_id=None,
    live_mirror_enabled=False,
    live_ctrader_account_id=None,
    live_lot_size=0.01,
    demo_lot_size=0.01,
    max_live_trades_day=3,
    learning_every_n_closes=3,
    min_lot=0.01,
    use_limit_entry=True,
    pending_entry_timeout_min=30,
    learning_daily_at_ny_end=True,
    confidence_threshold=45,
    include_history_in_decisions=False,
    orb_enabled=False,
    orb_range_minutes=20,
    orb_trade_window_minutes=90,
    orb_timeframe="5m",
    orb_confirmation="close",
    orb_require_retest=False,
    orb_retest_max_bars=3,
    orb_retest_tol_atr=0.15,
    orb_fakeout_filter=True,
    orb_break_buffer_atr=0.10,
    orb_break_buffer_range_pct=0.05,
    orb_min_break_body_atr=0.30,
    orb_min_range_atr=0.20,
    orb_max_range_atr=2.50,
    orb_sl_mode="range_opposite",
    orb_sl_atr_mult=1.2,
    orb_sl_range_buffer_atr=0.05,
    orb_tp_mode="range_multiple",
    orb_tp_range_mult=1.5,
    orb_tp_rr=1.5,
    orb_confidence_threshold=55,
    orb_max_calls_day=20,
    orb_min_global_calls_left=15,
    orb_max_trades_per_session=1,
    orb_entry_max_break_atr=0.60,
    orb_entry_max_break_range_pct=0.25,
)


def confidence_threshold() -> int:
    """Min confidence (0–100) required to fire a demo/live take."""
    raw = _env_int("GOLD_AI_CONFIDENCE_THRESHOLD", 45)
    return max(0, min(100, raw))


def env_defaults() -> GoldAiRuntimeConfig:
    uid = os.environ.get("GOLD_AI_TRADER_USER_ID", "").strip()
    demo_ctid = os.environ.get("GOLD_AI_TRADER_DEMO_ACCOUNT_ID", "").strip() or None
    return GoldAiRuntimeConfig(
        enabled=gold_ai_trader_enabled(),
        kill_switch=_env_bool("GOLD_AI_TRADER_KILL_SWITCH", False),
        london_start_hour=_env_int(
            "GOLD_AI_TRADER_LONDON_START", _SHARED_SESSION_HOURS["london"]["start_hour"]
        ),
        london_end_hour=_env_int(
            "GOLD_AI_TRADER_LONDON_END", _SHARED_SESSION_HOURS["london"]["end_hour"]
        ),
        ny_start_hour=_env_int(
            "GOLD_AI_TRADER_NY_START", _SHARED_SESSION_HOURS["new_york"]["start_hour"]
        ),
        ny_end_hour=_env_int(
            "GOLD_AI_TRADER_NY_END", _SHARED_SESSION_HOURS["new_york"]["end_hour"]
        ),
        max_calls_day=_env_int("GOLD_AI_TRADER_MAX_CALLS_DAY", 70),
        max_trades_day=_env_int("GOLD_AI_TRADER_MAX_TRADES_DAY", 6),
        no_overnight=_env_bool("GOLD_AI_TRADER_NO_OVERNIGHT", True),
        scan_interval_s=_env_float("GOLD_AI_TRADER_SCAN_INTERVAL_S", 20.0),
        model=os.environ.get("GOLD_AI_TRADER_MODEL", "claude-opus-4-8"),
        demo_user_id=int(uid) if uid.isdigit() else None,
        demo_ctrader_account_id=demo_ctid,
        live_mirror_enabled=False,
        live_ctrader_account_id=os.environ.get("GOLD_AI_TRADER_LIVE_ACCOUNT_ID", "").strip() or None,
        live_lot_size=_env_float("GOLD_AI_TRADER_LIVE_LOT", 0.01),
        demo_lot_size=_env_float("GOLD_AI_TRADER_DEMO_LOT", 0.01),
        max_live_trades_day=_env_int("GOLD_AI_TRADER_MAX_LIVE_TRADES_DAY", 3),
        learning_every_n_closes=_env_int("GOLD_AI_TRADER_LEARN_EVERY_N", 3),
        min_lot=_env_float("GOLD_AI_TRADER_MIN_LOT", 0.01),
        use_limit_entry=_env_bool("GOLD_AI_TRADER_USE_LIMIT_ENTRY", True),
        pending_entry_timeout_min=_env_int("GOLD_AI_TRADER_PENDING_TIMEOUT_MIN", 30),
        learning_daily_at_ny_end=_env_bool("GOLD_AI_TRADER_LEARNING_DAILY", True),
        confidence_threshold=confidence_threshold(),
        include_history_in_decisions=_env_bool(
            "GOLD_AI_INCLUDE_HISTORY_IN_DECISIONS", False
        ),
        orb_enabled=_env_bool("GOLD_AI_ORB_ENABLED", False),
        orb_range_minutes=_env_int("GOLD_AI_ORB_RANGE_MINUTES", 20),
        orb_trade_window_minutes=_env_int("GOLD_AI_ORB_TRADE_WINDOW_MINUTES", 90),
        orb_timeframe=(os.environ.get("GOLD_AI_ORB_TIMEFRAME", "5m") or "5m").strip().lower(),
        orb_confirmation=(os.environ.get("GOLD_AI_ORB_CONFIRMATION", "close") or "close").strip().lower(),
        orb_require_retest=_env_bool("GOLD_AI_ORB_REQUIRE_RETEST", False),
        orb_retest_max_bars=_env_int("GOLD_AI_ORB_RETEST_MAX_BARS", 3),
        orb_retest_tol_atr=_env_float("GOLD_AI_ORB_RETEST_TOL_ATR", 0.15),
        orb_fakeout_filter=_env_bool("GOLD_AI_ORB_FAKEOUT_FILTER", True),
        orb_break_buffer_atr=_env_float("GOLD_AI_ORB_BREAK_BUFFER_ATR", 0.10),
        orb_break_buffer_range_pct=_env_float("GOLD_AI_ORB_BREAK_BUFFER_RANGE_PCT", 0.05),
        orb_min_break_body_atr=_env_float("GOLD_AI_ORB_MIN_BREAK_BODY_ATR", 0.30),
        orb_min_range_atr=_env_float("GOLD_AI_ORB_MIN_RANGE_ATR", 0.20),
        orb_max_range_atr=_env_float("GOLD_AI_ORB_MAX_RANGE_ATR", 2.50),
        orb_sl_mode=(os.environ.get("GOLD_AI_ORB_SL_MODE", "range_opposite") or "range_opposite").strip().lower(),
        orb_sl_atr_mult=_env_float("GOLD_AI_ORB_SL_ATR_MULT", 1.2),
        orb_sl_range_buffer_atr=_env_float("GOLD_AI_ORB_SL_RANGE_BUFFER_ATR", 0.05),
        orb_tp_mode=(os.environ.get("GOLD_AI_ORB_TP_MODE", "range_multiple") or "range_multiple").strip().lower(),
        orb_tp_range_mult=_env_float("GOLD_AI_ORB_TP_RANGE_MULT", 1.5),
        orb_tp_rr=_env_float("GOLD_AI_ORB_TP_RR", 1.5),
        orb_confidence_threshold=_env_int("GOLD_AI_ORB_CONFIDENCE_THRESHOLD", 55),
        orb_max_calls_day=_env_int("GOLD_AI_ORB_MAX_CALLS_DAY", 20),
        orb_min_global_calls_left=_env_int("GOLD_AI_ORB_MIN_GLOBAL_CALLS_LEFT", 15),
        orb_max_trades_per_session=_env_int("GOLD_AI_ORB_MAX_TRADES_PER_SESSION", 1),
        orb_entry_max_break_atr=_env_float("GOLD_AI_ORB_ENTRY_MAX_BREAK_ATR", 0.60),
        orb_entry_max_break_range_pct=_env_float("GOLD_AI_ORB_ENTRY_MAX_BREAK_RANGE_PCT", 0.25),
    )


# Claude list pricing (USD per million tokens) — for cost estimates only.
# Gold AI defaults to Opus unless overridden via GOLD_AI_TRADER_MODEL.
HAIKU_INPUT_USD_PER_M = 0.25
HAIKU_OUTPUT_USD_PER_M = 1.25
HAIKU_CACHE_READ_USD_PER_M = 0.025
HAIKU_CACHE_WRITE_USD_PER_M = 0.3125

OPUS_INPUT_USD_PER_M = 15.0
OPUS_OUTPUT_USD_PER_M = 75.0
OPUS_CACHE_READ_USD_PER_M = 1.50
OPUS_CACHE_WRITE_USD_PER_M = 18.75

SYMBOL = "XAUUSD"
ASSET_CLASS = "forex"
