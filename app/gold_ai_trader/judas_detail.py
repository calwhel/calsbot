"""Enrich Judas swing TA messages for Claude (zone, reclaim, displacement)."""
from __future__ import annotations

import re
from typing import List, Optional, Tuple


def _parse_prior_range(msg: str) -> Optional[Tuple[float, float]]:
    m = re.search(
        r"prior range\s+([\d.]+)\s*[–\-]\s*([\d.]+)",
        msg,
        re.I,
    )
    if not m:
        return None
    try:
        lo, hi = float(m.group(1)), float(m.group(2))
        if hi >= lo:
            return lo, hi
    except (TypeError, ValueError):
        pass
    return None


def _last_closed_body_atr(k5: List[list], atr: float) -> float:
    if not k5 or len(k5) < 2 or atr <= 0:
        return 0.0
    row = k5[-2]
    try:
        o, c = float(row[1]), float(row[4])
        return abs(c - o) / atr
    except (TypeError, ValueError, IndexError):
        return 0.0


def enrich_judas_detail(
    msg: str,
    *,
    direction: str,
    price: float,
    k5: List[list],
    atr: float,
) -> str:
    """
    Append parseable zone + reclaim + displacement so Claude can score entry quality.
    """
    parsed = _parse_prior_range(msg)
    if not parsed or atr <= 0 or price <= 0:
        return msg

    prior_l, prior_h = parsed
    buf = 0.08 * atr
    d = (direction or "").upper()

    if d == "LONG":
        reclaim = prior_l
        zone_bot = prior_l - buf
        zone_top = prior_l + max(buf * 2, 0.12 * atr)
    else:
        reclaim = prior_h
        zone_bot = prior_h - max(buf * 2, 0.12 * atr)
        zone_top = prior_h + buf

    body_atr = _last_closed_body_atr(k5, atr)
    in_zone = zone_bot <= price <= zone_top
    zone_tag = "IN ZONE" if in_zone else "outside zone"

    return (
        f"{msg} | zone {zone_bot:.2f}–{zone_top:.2f} "
        f"reclaim @ {reclaim:.2f} spot={price:.2f} {zone_tag} "
        f"disp={body_atr:.2f}×ATR"
    )
