"""Detect FVG / IFVG / OB zones on OHLC bars for chart overlays."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from app.services.strategy_ta import _closes, _detect_fvg, _highs, _lows, _opens


@dataclass(frozen=True)
class ChartZone:
    kind: str  # fvg | ifvg | ob
    side: str  # bull | bear
    top: float
    bottom: float
    bar_idx: int
    label: str
    filled: bool = False

    def contains_price(self, price: float) -> bool:
        return self.bottom <= price <= self.top

    def distance_to_price(self, price: float) -> float:
        if self.contains_price(price):
            return 0.0
        if price > self.top:
            return price - self.top
        return self.bottom - price


def _zone_near_price(zone: ChartZone, spot: float, *, max_dist_pct: float = 0.004) -> bool:
    if spot <= 0:
        return True
    if zone.contains_price(spot):
        return True
    return zone.distance_to_price(spot) / spot <= max_dist_pct


def _pick_zones(candidates: List[ChartZone], spot: float, *, max_zones: int) -> List[ChartZone]:
    if not candidates:
        return []
    near = [z for z in candidates if _zone_near_price(z, spot)]
    pool = near if near else candidates
    pool.sort(key=lambda z: (z.distance_to_price(spot), z.bar_idx))
    return pool[:max(1, max_zones)]


def _fvg_zones(bars: Sequence[Sequence[float]], spot: float, *, tf: str) -> List[ChartZone]:
    if len(bars) < 10:
        return []
    gaps = _detect_fvg(
        list(bars),
        min_gap_usd=0.15,
        compute_quality=True,
        only_unfilled=False,
    )
    out: List[ChartZone] = []
    for g in gaps:
        if int(g.get("age") or 0) > 60:
            continue
        side = str(g.get("type") or "bullish")
        bull = side.startswith("bull")
        filled = bool(g.get("filled"))
        kind = "ifvg" if filled else "fvg"
        bottom = float(g["bottom"])
        top = float(g["top"])
        idx = int(g.get("idx") or 0)
        tag = "IFVG" if filled else "FVG"
        side_lbl = "bull" if bull else "bear"
        label = f"{tag} {side_lbl} {bottom:.2f}-{top:.2f}"
        out.append(
            ChartZone(
                kind=kind,
                side=side_lbl,
                top=top,
                bottom=bottom,
                bar_idx=idx,
                label=label,
                filled=filled,
            )
        )
    return _pick_zones(out, spot, max_zones=4)


def _ob_zones(bars: Sequence[Sequence[float]], spot: float, *, tf: str) -> List[ChartZone]:
    klines = list(bars)
    n = len(klines)
    if n < 12:
        return []
    closes = _closes(klines)
    opens = _opens(klines)
    highs = _highs(klines)
    lows = _lows(klines)
    out: List[ChartZone] = []

    def _try_bull(i: int) -> Optional[ChartZone]:
        if i < 5 or i >= n - 2:
            return None
        if not (opens[i] > closes[i]):
            return None
        nxt = range(i + 1, min(i + 3, n))
        if len(nxt) < 2 or not all(closes[j] > opens[j] for j in nxt):
            return None
        ob_high = float(highs[i])
        ob_low = float(lows[i])
        return ChartZone(
            kind="ob",
            side="bull",
            top=ob_high,
            bottom=ob_low,
            bar_idx=i,
            label=f"OB bull {ob_low:.2f}-{ob_high:.2f}",
        )

    def _try_bear(i: int) -> Optional[ChartZone]:
        if i < 5 or i >= n - 2:
            return None
        if not (closes[i] > opens[i]):
            return None
        nxt = range(i + 1, min(i + 3, n))
        if len(nxt) < 2 or not all(closes[j] < opens[j] for j in nxt):
            return None
        ob_high = float(highs[i])
        ob_low = float(lows[i])
        return ChartZone(
            kind="ob",
            side="bear",
            top=ob_high,
            bottom=ob_low,
            bar_idx=i,
            label=f"OB bear {ob_low:.2f}-{ob_high:.2f}",
        )

    for i in range(n - 3, 5, -1):
        z = _try_bull(i)
        if z:
            out.append(z)
            break
    for i in range(n - 3, 5, -1):
        z = _try_bear(i)
        if z:
            out.append(z)
            break
    return _pick_zones(out, spot, max_zones=2)


def detect_chart_zones(
    bars: Sequence[Sequence[float]],
    spot: float,
    *,
    timeframe: str = "5m",
    include_ob: bool = True,
) -> List[ChartZone]:
    """Return zones to draw on a chart (nearest unfilled FVG + recent OB)."""
    zones = _fvg_zones(bars, spot, tf=timeframe)
    if include_ob:
        zones.extend(_ob_zones(bars, spot, tf=timeframe))
    # De-dupe overlapping labels — prefer lower distance to spot
    zones.sort(key=lambda z: (z.distance_to_price(spot), -z.bar_idx))
    seen: set[tuple] = set()
    unique: List[ChartZone] = []
    for z in zones:
        key = (z.kind, round(z.bottom, 2), round(z.top, 2))
        if key in seen:
            continue
        seen.add(key)
        unique.append(z)
    return unique[:6]


def format_zones_for_prompt(
    zones_by_tf: dict[str, List[ChartZone]],
    spot: float,
) -> str:
    """Text summary of detected zones for Gemini (cross-check with shaded charts)."""
    lines = ["DETECTED ZONES (engine — shaded on 5m/15m charts):"]
    if spot > 0:
        lines.append(f"  spot={spot:.2f}")
    any_zone = False
    for tf in ("5m", "15m"):
        zones = zones_by_tf.get(tf) or []
        if not zones:
            continue
        any_zone = True
        lines.append(f"  {tf}:")
        for z in zones:
            in_zone = "IN" if z.contains_price(spot) else "near"
            lines.append(f"    - {z.label} ({in_zone}, bar {z.bar_idx})")
    if not any_zone:
        lines.append("  (no FVG/IFVG/OB zones detected near price on 5m/15m)")
    return "\n".join(lines)
