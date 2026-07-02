"""Environment + runtime configuration for Gemini Vision Gold Trader."""
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


SYMBOL = "XAUUSD"
ASSET_CLASS = "forex"


def gemini_gold_enabled() -> bool:
    return _env_bool("GEMINI_GOLD_ENABLED", False)


def gemini_gold_dry_run() -> bool:
    return _env_bool("GEMINI_GOLD_DRY_RUN", True)


def is_standalone_gemini_gold() -> bool:
    return _env_bool("GEMINI_GOLD_STANDALONE", False)


def gemini_gold_loop_disabled_in_gunicorn() -> bool:
    return _env_bool("DISABLE_GEMINI_GOLD_IN_GUNICORN", True)


@dataclass
class GeminiGoldRuntimeConfig:
    enabled: bool
    kill_switch: bool
    dry_run: bool
    max_calls_day: int
    max_trades_day: int
    scan_interval_s: float
    model: str
    demo_user_id: Optional[int]
    demo_ctrader_account_id: Optional[str]
    demo_lot_size: float
    confidence_threshold: int
    chart_bars: int
    min_sl_pips: float
    entry_max_drift_pct: float
    min_trade_gap_min: int


def env_defaults() -> GeminiGoldRuntimeConfig:
    demo_uid_raw = os.environ.get("GEMINI_GOLD_USER_ID", "").strip()
    demo_uid: Optional[int] = None
    if demo_uid_raw:
        try:
            demo_uid = int(demo_uid_raw)
        except ValueError:
            demo_uid = None

    demo_ctid = os.environ.get("GEMINI_GOLD_DEMO_ACCOUNT_ID", "").strip() or None

    return GeminiGoldRuntimeConfig(
        enabled=gemini_gold_enabled(),
        kill_switch=_env_bool("GEMINI_GOLD_KILL_SWITCH", False),
        dry_run=gemini_gold_dry_run(),
        max_calls_day=_env_int("GEMINI_GOLD_MAX_CALLS_DAY", 340),
        max_trades_day=_env_int("GEMINI_GOLD_MAX_TRADES_DAY", 4),
        scan_interval_s=max(60.0, _env_float("GEMINI_GOLD_SCAN_INTERVAL_S", 180.0)),
        model=os.environ.get("GEMINI_GOLD_MODEL", "gemini-2.5-flash").strip()
        or "gemini-2.5-flash",
        demo_user_id=demo_uid,
        demo_ctrader_account_id=demo_ctid,
        demo_lot_size=max(0.01, _env_float("GEMINI_GOLD_DEMO_LOT", 0.01)),
        confidence_threshold=_env_int("GEMINI_GOLD_CONFIDENCE_THRESHOLD", 90),
        chart_bars=max(20, _env_int("GEMINI_GOLD_CHART_BARS", 80)),
        min_sl_pips=_env_float("GEMINI_GOLD_MIN_SL_PIPS", 60.0),
        entry_max_drift_pct=_env_float("GEMINI_GOLD_ENTRY_MAX_DRIFT_PCT", 0.15),
        min_trade_gap_min=max(0, _env_int("GEMINI_GOLD_MIN_TRADE_GAP_MIN", 20)),
    )
