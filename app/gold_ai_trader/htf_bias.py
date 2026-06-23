"""HTF bias helpers for Gold AI Trader setup alignment and context."""
from __future__ import annotations

from typing import List, Optional, Tuple

from app.gold_ai_trader.context_regime import compute_1h_trend, _closes, _structure_label


def compute_4h_trend(k_4h: List[list]) -> str:
    """EMA8/21 + slope on 4h — mirrors 1h trend logic."""
    closes = _closes(k_4h)
    if len(closes) < 10:
        return "unavailable"

    def _ema(values: List[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        k = 2 / (period + 1)
        ema = sum(values[:period]) / period
        for v in values[period:]:
            ema = v * k + ema * (1 - k)
        return ema

    ema_fast = _ema(closes, 8)
    ema_slow = _ema(closes, 21)
    if ema_fast is None or ema_slow is None:
        return "mixed/ranging"
    slope = closes[-1] - closes[-5] if len(closes) >= 5 else 0.0
    if ema_fast > ema_slow * 1.0003 and slope > 0:
        return "bullish"
    if ema_fast < ema_slow * 0.9997 and slope < 0:
        return "bearish"
    return "mixed/ranging"


def daily_range(k_daily: List[list]) -> Tuple[Optional[float], Optional[float]]:
    """Today's daily high/low from most recent daily candle."""
    if not k_daily:
        return None, None
    row = k_daily[-1]
    if not row or len(row) < 5:
        return None, None
    try:
        return float(row[2]), float(row[3])
    except (TypeError, ValueError):
        return None, None


def htf_bias_summary(
    k_1h: List[list],
    k_4h: List[list],
    k_daily: List[list],
) -> dict:
    """Structured HTF bias for Claude context and setup alignment."""
    trend_1h = compute_1h_trend(k_1h)
    trend_4h = compute_4h_trend(k_4h)
    structure_1h = _structure_label(k_1h)
    structure_4h = _structure_label(k_4h) if k_4h else "unavailable"
    d_hi, d_lo = daily_range(k_daily)

    aligned = "mixed"
    if trend_1h == trend_4h == "bullish":
        aligned = "bullish"
    elif trend_1h == trend_4h == "bearish":
        aligned = "bearish"

    return {
        "trend_1h": trend_1h,
        "trend_4h": trend_4h,
        "structure_1h": structure_1h,
        "structure_4h": structure_4h,
        "htf_bias": aligned,
        "daily_high": d_hi,
        "daily_low": d_lo,
    }


def direction_aligns_with_htf(direction: str, bias: dict) -> Tuple[bool, str]:
    """Check whether setup direction aligns with consolidated HTF bias."""
    htf = (bias.get("htf_bias") or "mixed").lower()
    d = (direction or "").upper()
    if htf == "mixed":
        return True, "htf_mixed_allowed"
    if d == "LONG" and htf == "bullish":
        return True, "htf_aligned_bull"
    if d == "SHORT" and htf == "bearish":
        return True, "htf_aligned_bear"
    if d == "LONG" and htf == "bearish":
        return False, "counter_htf_bear"
    if d == "SHORT" and htf == "bullish":
        return False, "counter_htf_bull"
    return True, "htf_neutral"
