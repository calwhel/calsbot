"""Environment + runtime configuration for Gemini Vision Gold Trader."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


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


EXECUTION_MODE_DEMO = "demo"
EXECUTION_MODE_LIVE = "live"


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
    execution_mode: str
    live_ctrader_account_id: Optional[str]
    live_lot_size: float
    live_mirror_enabled: bool
    max_live_trades_day: int
    max_open_positions: int
    confidence_threshold: int
    use_limit_entry: bool
    pending_entry_timeout_min: int
    orb_enabled: bool
    orb_confidence_threshold: int
    orb_max_calls_day: int
    orb_max_trades_per_session: int
    trade_sessions: Tuple[str, ...]
    custom_trade_hours_enabled: bool
    trade_hours_start_utc: str
    trade_hours_end_utc: str
    chart_bars: int
    chart_bars_1m: int
    min_sl_pips: float
    max_sl_pips: float
    min_rr: float
    max_rr: float
    entry_max_drift_pct: float
    min_trade_gap_min: int
    two_step_scan: bool


def gemini_gold_two_step_scan() -> bool:
    return _env_bool("GEMINI_GOLD_TWO_STEP_SCAN", True)


GEMINI_GOLD_REVIEW_MODEL_DEFAULT = "gemini-3.1-pro-preview"

# Google shut down gemini-2.5-pro early on the consumer API — remap legacy env values.
_DEPRECATED_GEMINI_REVIEW_MODELS = {
    "gemini-2.5-pro": GEMINI_GOLD_REVIEW_MODEL_DEFAULT,
    "gemini-2.5-pro-preview-03-25": GEMINI_GOLD_REVIEW_MODEL_DEFAULT,
    "gemini-2.5-pro-preview-05-06": GEMINI_GOLD_REVIEW_MODEL_DEFAULT,
    "gemini-2.5-pro-preview-06-05": GEMINI_GOLD_REVIEW_MODEL_DEFAULT,
    "gemini-3-pro-preview": GEMINI_GOLD_REVIEW_MODEL_DEFAULT,
}


def gemini_gold_review_model() -> str:
    raw = (
        os.environ.get("GEMINI_GOLD_REVIEW_MODEL", GEMINI_GOLD_REVIEW_MODEL_DEFAULT).strip()
        or GEMINI_GOLD_REVIEW_MODEL_DEFAULT
    )
    return _DEPRECATED_GEMINI_REVIEW_MODELS.get(raw, raw)


def env_defaults() -> GeminiGoldRuntimeConfig:
    demo_uid_raw = os.environ.get("GEMINI_GOLD_USER_ID", "").strip()
    demo_uid: Optional[int] = None
    if demo_uid_raw:
        try:
            demo_uid = int(demo_uid_raw)
        except ValueError:
            demo_uid = None

    demo_ctid = os.environ.get("GEMINI_GOLD_DEMO_ACCOUNT_ID", "").strip() or None
    live_ctid = os.environ.get("GEMINI_GOLD_LIVE_ACCOUNT_ID", "").strip() or None
    exec_mode = os.environ.get("GEMINI_GOLD_EXECUTION_MODE", EXECUTION_MODE_DEMO).strip().lower()
    if exec_mode not in (EXECUTION_MODE_DEMO, EXECUTION_MODE_LIVE):
        exec_mode = EXECUTION_MODE_DEMO

    return GeminiGoldRuntimeConfig(
        enabled=gemini_gold_enabled(),
        kill_switch=_env_bool("GEMINI_GOLD_KILL_SWITCH", False),
        dry_run=gemini_gold_dry_run(),
        max_calls_day=_env_int("GEMINI_GOLD_MAX_CALLS_DAY", 340),
        max_trades_day=_env_int("GEMINI_GOLD_MAX_TRADES_DAY", 4),
        scan_interval_s=max(60.0, _env_float("GEMINI_GOLD_SCAN_INTERVAL_S", 120.0)),
        model=os.environ.get("GEMINI_GOLD_MODEL", "gemini-2.5-flash").strip()
        or "gemini-2.5-flash",
        demo_user_id=demo_uid,
        demo_ctrader_account_id=demo_ctid,
        demo_lot_size=max(0.01, _env_float("GEMINI_GOLD_DEMO_LOT", 0.01)),
        execution_mode=exec_mode,
        live_ctrader_account_id=live_ctid,
        live_lot_size=max(0.01, _env_float("GEMINI_GOLD_LIVE_LOT", 0.01)),
        live_mirror_enabled=False,
        max_live_trades_day=max(1, _env_int("GEMINI_GOLD_MAX_LIVE_TRADES_DAY", 3)),
        # 0 (or negative) = no concurrent-open-position cap (off). The daily
        # trade cap (max_trades_day) still bounds total activity per day.
        max_open_positions=max(0, _env_int("GEMINI_GOLD_MAX_OPEN_POSITIONS", 0)),
        confidence_threshold=_env_int("GEMINI_GOLD_CONFIDENCE_THRESHOLD", 85),
        use_limit_entry=_env_bool("GEMINI_GOLD_USE_LIMIT_ENTRY", False),
        pending_entry_timeout_min=max(5, _env_int("GEMINI_GOLD_PENDING_TIMEOUT_MIN", 30)),
        orb_enabled=_env_bool("GEMINI_GOLD_ORB_ENABLED", False),
        orb_confidence_threshold=_env_int("GEMINI_GOLD_ORB_CONFIDENCE_THRESHOLD", 65),
        orb_max_calls_day=_env_int("GEMINI_GOLD_ORB_MAX_CALLS_DAY", 20),
        orb_max_trades_per_session=_env_int("GEMINI_GOLD_ORB_MAX_TRADES_PER_SESSION", 1),
        trade_sessions=("asia", "london", "new_york"),
        custom_trade_hours_enabled=_env_bool("GEMINI_GOLD_CUSTOM_TRADE_HOURS", False),
        trade_hours_start_utc=os.environ.get("GEMINI_GOLD_TRADE_HOURS_START_UTC", "12:00").strip()
        or "12:00",
        trade_hours_end_utc=os.environ.get("GEMINI_GOLD_TRADE_HOURS_END_UTC", "21:00").strip()
        or "21:00",
        chart_bars=max(20, _env_int("GEMINI_GOLD_CHART_BARS", 80)),
        chart_bars_1m=max(20, _env_int("GEMINI_GOLD_CHART_BARS_1M", 60)),
        min_sl_pips=_env_float("GEMINI_GOLD_MIN_SL_PIPS", 10.0),
        max_sl_pips=_env_float("GEMINI_GOLD_MAX_SL_PIPS", 150.0),
        min_rr=max(1.0, _env_float("GEMINI_GOLD_MIN_RR", 1.0)),
        max_rr=min(2.0, max(1.0, _env_float("GEMINI_GOLD_MAX_RR", 2.0))),
        entry_max_drift_pct=_env_float("GEMINI_GOLD_ENTRY_MAX_DRIFT_PCT", 0.25),
        min_trade_gap_min=max(0, _env_int("GEMINI_GOLD_MIN_TRADE_GAP_MIN", 20)),
        two_step_scan=gemini_gold_two_step_scan(),
    )
