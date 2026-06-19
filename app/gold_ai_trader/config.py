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


DEFAULTS = GoldAiRuntimeConfig(
    enabled=False,
    kill_switch=False,
    london_start_hour=7,
    london_end_hour=10,
    ny_start_hour=13,
    ny_end_hour=16,
    max_calls_day=22,
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
)


def env_defaults() -> GoldAiRuntimeConfig:
    uid = os.environ.get("GOLD_AI_TRADER_USER_ID", "").strip()
    demo_ctid = os.environ.get("GOLD_AI_TRADER_DEMO_ACCOUNT_ID", "").strip() or None
    return GoldAiRuntimeConfig(
        enabled=gold_ai_trader_enabled(),
        kill_switch=_env_bool("GOLD_AI_TRADER_KILL_SWITCH", False),
        london_start_hour=_env_int("GOLD_AI_TRADER_LONDON_START", 7),
        london_end_hour=_env_int("GOLD_AI_TRADER_LONDON_END", 10),
        ny_start_hour=_env_int("GOLD_AI_TRADER_NY_START", 13),
        ny_end_hour=_env_int("GOLD_AI_TRADER_NY_END", 16),
        max_calls_day=_env_int("GOLD_AI_TRADER_MAX_CALLS_DAY", 22),
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
    )


# Opus 4.8 list pricing (USD per million tokens) — for cost estimates only.
OPUS_INPUT_USD_PER_M = 15.0
OPUS_OUTPUT_USD_PER_M = 75.0
OPUS_CACHE_READ_USD_PER_M = 1.50
OPUS_CACHE_WRITE_USD_PER_M = 18.75

SYMBOL = "XAUUSD"
ASSET_CLASS = "forex"
