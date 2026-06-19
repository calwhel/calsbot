"""Higher-timeframe regime read for Gold AI Trader context."""
from __future__ import annotations

from typing import List, Optional, Tuple


def _closes(rows: List[list]) -> List[float]:
    out = []
    for row in rows or []:
        if row and len(row) >= 5:
            try:
                out.append(float(row[4]))
            except (TypeError, ValueError):
                continue
    return out


def _ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = v * k + ema * (1 - k)
    return ema


def _atr_from_rows(rows: List[list], period: int = 14) -> float:
    closes = _closes(rows)
    if len(closes) < period + 1:
        return 0.0
    trs = [abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))]
    if len(trs) < period:
        return sum(trs) / max(len(trs), 1)
    return sum(trs[-period:]) / period


def _structure_label(rows: List[list]) -> str:
    """HH/HL vs LH/LL over last 8 bars."""
    if len(rows) < 6:
        return "mixed/ranging"
    tail = rows[-8:]
    highs = [float(r[2]) for r in tail if r and len(r) >= 3]
    lows = [float(r[3]) for r in tail if r and len(r) >= 3]
    if len(highs) < 4 or len(lows) < 4:
        return "mixed/ranging"
    mid = len(highs) // 2
    h1, h2 = max(highs[:mid]), max(highs[mid:])
    l1, l2 = min(lows[:mid]), min(lows[mid:])
    if h2 > h1 and l2 > l1:
        return "trending"
    if h2 < h1 and l2 < l1:
        return "trending"
    overlap = min(h1, h2) - max(l1, l2)
    rng = max(max(highs) - min(lows), 1e-9)
    if overlap / rng > 0.35:
        return "ranging"
    return "mixed/ranging"


def compute_1h_trend(k_1h: List[list]) -> str:
    closes = _closes(k_1h)
    if len(closes) < 10:
        return "unavailable"
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


def compute_volatility_label(k_5m: List[list]) -> Tuple[str, Optional[float]]:
    closes = _closes(k_5m)
    if len(closes) < 20:
        return "unavailable", None
    atr_now = _atr_from_rows(k_5m[-20:], 14)
    # rolling ATR samples
    samples = []
    for i in range(20, min(len(k_5m), 55)):
        chunk = k_5m[i - 20 : i]
        a = _atr_from_rows(chunk, 14)
        if a > 0:
            samples.append(a)
    if not samples or atr_now <= 0:
        if atr_now <= 0:
            return "unavailable", None
        return "normal", 1.0
    avg = sum(samples) / len(samples)
    ratio = atr_now / avg if avg > 0 else 1.0
    if ratio >= 1.3:
        return "high", ratio
    if ratio <= 0.75:
        return "low", ratio
    return "normal", ratio


def build_regime_block(k_1h: List[list], k_5m: List[list]) -> List[str]:
    if not k_1h or len(k_1h) < 10:
        return ["=== REGIME ===", "Regime: unavailable (insufficient 1h data)"]

    trend = compute_1h_trend(k_1h)
    vol_label, vol_ratio = compute_volatility_label(k_5m)
    structure = _structure_label(k_1h)

    if trend == "unavailable" or vol_label == "unavailable":
        return ["=== REGIME ===", "Regime: unavailable"]

    vol_txt = vol_label
    if vol_ratio is not None:
        vol_txt = f"{vol_label} ({vol_ratio:.1f}× avg ATR)"

    note = "follow-with-trend favored; fading needs strong displacement confirmation."
    if structure == "ranging":
        note = "range conditions — fade edges with reclaim; avoid chasing mid-range."
    elif vol_label == "high":
        note = "elevated vol — widen invalidation awareness; require cleaner triggers."

    return [
        "=== REGIME ===",
        f"1h trend: {trend} | Volatility: {vol_txt} | Structure: {structure}",
        f"Note: {note}",
    ]
