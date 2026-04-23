"""
Liquidity Wall Scanner — multi-exchange order book aggregator.

Pulls live order books from 3 free public exchanges (Bybit, OKX, MEXC),
normalizes them, finds the biggest resting buy/sell walls within configurable
distance bands of current price, scores cross-exchange confidence, and returns
a trader-friendly summary plus an AI paragraph.

Used by the Telegram bot's /walls command.

NOTE: Binance is omitted because Replit IPs are geoblocked (HTTP 451).
WebSocket live order-book maintenance and spoof-detection-with-persistence
require a long-running background worker — left as a follow-up.
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_MAX_BAND_PCT = 5.0
DEFAULT_MIN_DISTANCE_PCT = 0.20   # skip orders inside this — that's spread, not a wall
DEFAULT_TOP_N = 4
DEFAULT_MIN_NOTIONAL_USD = 100_000
# Pick the biggest wall in each of these distance bands so the top zones are spread across the chart
# (one immediate, one mid-pullback, one swing zone, one far-out massive wall). Edges in % from mid.
DEFAULT_DISTANCE_BANDS = [(0.20, 0.60), (0.60, 1.50), (1.50, 3.00), (3.00, 5.00)]
HTTP_TIMEOUT = 6.0


@dataclass
class Wall:
    price: float
    size_native: float
    size_usd: float
    distance_pct: float           # signed: positive = above price (sell side), negative = below
    side: str                      # "buy" or "sell"
    exchanges: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> str:
        n = len(self.exchanges)
        if n >= 3:
            return "confirmed across multiple exchanges"
        if n == 2:
            return "seen on two exchanges"
        return "seen mainly on one exchange"


@dataclass
class WallReport:
    symbol: str
    price: float
    biggest_buy: Optional[Wall]
    biggest_sell: Optional[Wall]
    top_buys: list[Wall]
    top_sells: list[Wall]
    pressure_label: str            # "Strongly bullish" / "Slightly bullish" / "Neutral" / "Slightly bearish" / "Strongly bearish"
    pressure_score: float          # -1.0 to +1.0
    exchanges_used: list[str]
    exchanges_failed: list[str]
    ai_summary: str
    best_zone_to_watch: str
    # Enrichment context (added in v2)
    volume_24h_usd: float = 0.0
    volume_1h_usd: float = 0.0
    liq_24h_long_usd: float = 0.0
    liq_24h_short_usd: float = 0.0
    wall_behavior: dict = field(default_factory=dict)   # rounded-price → "FRESH"/"GROWING"/...
    previous_snapshot_age_secs: Optional[int] = None
    is_being_watched: bool = False                      # set by /walls handler if user has watch active


# ───────────────────────── Symbol normalization ─────────────────────────

def _norm_symbol(raw: str) -> str:
    """Normalize user input → BASEUSDT (uppercase, no separators)."""
    s = (raw or "").upper().strip().replace("/", "").replace("-", "").replace("_", "")
    if not s.endswith("USDT") and not s.endswith("USD"):
        s = f"{s}USDT"
    return s


def _split_base(symbol: str) -> tuple[str, str]:
    if symbol.endswith("USDT"):
        return symbol[:-4], "USDT"
    if symbol.endswith("USD"):
        return symbol[:-3], "USD"
    return symbol, ""


# ───────────────────────── Exchange adapters ─────────────────────────

async def _fetch_bybit(client: httpx.AsyncClient, symbol: str) -> Optional[dict]:
    """Bybit SPOT order book — sizes in native base coin units."""
    try:
        r = await client.get(
            "https://api.bybit.com/v5/market/orderbook",
            params={"category": "spot", "symbol": symbol, "limit": 200},
        )
        r.raise_for_status()
        d = r.json()
        if d.get("retCode") != 0:
            return None
        result = d.get("result") or {}
        bids = [(float(p), float(s)) for p, s in result.get("b", [])]
        asks = [(float(p), float(s)) for p, s in result.get("a", [])]
        if not bids or not asks:
            return None
        return {"bids": bids, "asks": asks, "exchange": "bybit"}
    except Exception as e:
        logger.debug(f"bybit fetch failed for {symbol}: {e}")
        return None


async def _fetch_okx(client: httpx.AsyncClient, symbol: str) -> Optional[dict]:
    """OKX SPOT order book — sizes in native base coin units (no contract conversion needed)."""
    base, quote = _split_base(symbol)
    inst_id = f"{base}-{quote}"
    try:
        r = await client.get(
            "https://www.okx.com/api/v5/market/books",
            params={"instId": inst_id, "sz": 400},
        )
        r.raise_for_status()
        d = r.json()
        if d.get("code") != "0":
            return None
        rows = (d.get("data") or [{}])[0]
        bids = [(float(p), float(s)) for p, s, *_ in rows.get("bids", [])]
        asks = [(float(p), float(s)) for p, s, *_ in rows.get("asks", [])]
        if not bids or not asks:
            return None
        return {"bids": bids, "asks": asks, "exchange": "okx"}
    except Exception as e:
        logger.debug(f"okx fetch failed for {symbol}: {e}")
        return None


async def _fetch_mexc(client: httpx.AsyncClient, symbol: str) -> Optional[dict]:
    """MEXC SPOT depth — sizes in native base coin units."""
    try:
        r = await client.get(
            "https://api.mexc.com/api/v3/depth",
            params={"symbol": symbol, "limit": 500},
        )
        r.raise_for_status()
        d = r.json()
        bids = [(float(p), float(s)) for p, s in d.get("bids", [])]
        asks = [(float(p), float(s)) for p, s in d.get("asks", [])]
        if not bids or not asks:
            return None
        return {"bids": bids, "asks": asks, "exchange": "mexc"}
    except Exception as e:
        logger.debug(f"mexc fetch failed for {symbol}: {e}")
        return None


async def _fetch_kraken(client: httpx.AsyncClient, symbol: str) -> Optional[dict]:
    """Kraken SPOT depth — sizes in native base coin units. Uses XBT for BTC."""
    base, quote = _split_base(symbol)
    # Kraken quirks: BTC=XBT, and pair format is BASEQUOTE (no separator) for the depth endpoint
    base_k = "XBT" if base == "BTC" else base
    pair = f"{base_k}{quote}"
    try:
        r = await client.get(
            "https://api.kraken.com/0/public/Depth",
            params={"pair": pair, "count": 200},
        )
        r.raise_for_status()
        d = r.json()
        if d.get("error"):
            return None
        result = d.get("result") or {}
        if not result:
            return None
        # Kraken nests under the resolved pair name (which may differ from request)
        first_pair_data = next(iter(result.values()))
        bids = [(float(p), float(s)) for p, s, *_ in first_pair_data.get("bids", [])]
        asks = [(float(p), float(s)) for p, s, *_ in first_pair_data.get("asks", [])]
        if not bids or not asks:
            return None
        return {"bids": bids, "asks": asks, "exchange": "kraken"}
    except Exception as e:
        logger.debug(f"kraken fetch failed for {symbol}: {e}")
        return None


async def _fetch_all_books(symbol: str) -> tuple[list[dict], list[str]]:
    """Fetch from all exchanges in parallel. Returns (successful_books, failed_exchange_names)."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        results = await asyncio.gather(
            _fetch_bybit(client, symbol),
            _fetch_okx(client, symbol),
            _fetch_mexc(client, symbol),
            _fetch_kraken(client, symbol),
            return_exceptions=True,
        )
    books, failed = [], []
    for name, res in zip(["bybit", "okx", "mexc", "kraken"], results):
        if isinstance(res, dict):
            books.append(res)
        else:
            failed.append(name)
    return books, failed


# ───────────────────────── Wall detection ─────────────────────────

def _mid_price(books: list[dict]) -> float:
    """Best mid price across exchanges = average of (best_bid + best_ask)/2."""
    mids = []
    for b in books:
        if b["bids"] and b["asks"]:
            mids.append((b["bids"][0][0] + b["asks"][0][0]) / 2.0)
    return sum(mids) / len(mids) if mids else 0.0


def _bucket_size(price: float) -> float:
    """Pick a price-bucket size that groups nearby orders into a 'wall'.
    Target ~0.10% of price — anything tighter just clusters adjacent ticks
    that aren't really separate walls."""
    target = price * 0.001
    if target <= 0:
        return 0.01
    # Round to a reasonable power-of-ten step
    import math
    mag = 10 ** math.floor(math.log10(target))
    for mult in (1, 2.5, 5, 10):
        step = mag * mult
        if step >= target:
            return step
    return target


def _aggregate_walls(
    books: list[dict],
    mid: float,
    side: str,                       # "buy" or "sell"
    max_band_pct: float,
    min_notional_usd: float,
    min_distance_pct: float = DEFAULT_MIN_DISTANCE_PCT,
) -> list[Wall]:
    """Group nearby orders into buckets (walls), aggregate across exchanges, score by USD size.
    Orders inside ±min_distance_pct of mid are treated as spread/noise and skipped."""
    bucket = _bucket_size(mid)
    # Map: bucket_price → {"size": float, "exchanges": set, "price_weighted": (sum_p*size, sum_size)}
    buckets: dict[float, dict] = {}

    for book in books:
        levels = book["bids"] if side == "buy" else book["asks"]
        for price, size in levels:
            if price <= 0 or size <= 0:
                continue
            distance_pct = (price - mid) / mid * 100.0
            if abs(distance_pct) > max_band_pct:
                continue
            if abs(distance_pct) < min_distance_pct:
                continue                          # too close to price — that's the spread, not a wall
            if side == "buy" and distance_pct > 0:
                continue
            if side == "sell" and distance_pct < 0:
                continue
            # Bucket by floored price
            key = round(price / bucket) * bucket
            entry = buckets.setdefault(key, {"size": 0.0, "ex": set(), "psum": 0.0, "ssum": 0.0})
            entry["size"] += size
            entry["ex"].add(book["exchange"])
            entry["psum"] += price * size
            entry["ssum"] += size

    walls: list[Wall] = []
    for key, e in buckets.items():
        avg_price = e["psum"] / e["ssum"] if e["ssum"] > 0 else key
        usd = e["size"] * avg_price
        if usd < min_notional_usd:
            continue
        walls.append(Wall(
            price=avg_price,
            size_native=e["size"],
            size_usd=usd,
            distance_pct=(avg_price - mid) / mid * 100.0,
            side=side,
            exchanges=sorted(e["ex"]),
        ))

    walls.sort(key=lambda w: w.size_usd, reverse=True)
    return walls


def _pick_band_leaders(walls: list[Wall], bands: list[tuple[float, float]]) -> list[Wall]:
    """From a sorted list of walls, return the BIGGEST wall in each distance band.
    Result is sorted by distance from price (closest first) so the output reads as a ladder.
    Bands that contain no qualifying wall are skipped."""
    chosen: list[Wall] = []
    for low, high in bands:
        candidates = [w for w in walls if low <= abs(w.distance_pct) < high]
        if not candidates:
            continue
        chosen.append(max(candidates, key=lambda w: w.size_usd))
    chosen.sort(key=lambda w: abs(w.distance_pct))
    return chosen


def _pressure(top_buys: list[Wall], top_sells: list[Wall]) -> tuple[str, float]:
    """Compare buy vs sell USD liquidity inside scan band, weighted by inverse distance."""
    def weight(w: Wall) -> float:
        return w.size_usd / max(abs(w.distance_pct), 0.05)

    buy_w = sum(weight(w) for w in top_buys)
    sell_w = sum(weight(w) for w in top_sells)
    total = buy_w + sell_w
    if total <= 0:
        return "Neutral", 0.0
    score = (buy_w - sell_w) / total          # -1 (all sells) → +1 (all buys)
    if score > 0.35:
        return "Strongly bullish", score
    if score > 0.12:
        return "Slightly bullish", score
    if score < -0.35:
        return "Strongly bearish", score
    if score < -0.12:
        return "Slightly bearish", score
    return "Neutral", score


# ───────────────────────── Output formatting ─────────────────────────

def _size_tier(usd: float) -> tuple[str, str]:
    """Tag a wall by USD notional. Returns (label, emoji).
    Tiers calibrated so 'big'+'huge' = walls actually worth fading/shorting into."""
    if usd >= 5_000_000:
        return "HUGE", "🐋"
    if usd >= 1_000_000:
        return "BIG", "🟦"
    if usd >= 250_000:
        return "MEDIUM", "🔹"
    return "small", "·"


def _fmt_usd(v: float) -> str:
    if v >= 1_000_000:
        return f"${v / 1_000_000:.2f}M"
    if v >= 1_000:
        return f"${v / 1_000:.0f}k"
    return f"${v:.0f}"


def _fmt_price(p: float) -> str:
    if p >= 1000:
        return f"{p:,.2f}"
    if p >= 1:
        return f"{p:,.4f}"
    if p >= 0.01:
        return f"{p:.5f}"
    return f"{p:.8f}"


def _fmt_dist(pct: float) -> str:
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.2f}%"


def _fallback_summary(report: dict) -> str:
    """Plain-English summary if the AI call fails."""
    parts = []
    bb = report["biggest_buy"]
    bs = report["biggest_sell"]
    if bb:
        parts.append(f"Buyers are stacked at {_fmt_price(bb.price)} ({_fmt_usd(bb.size_usd)} {bb.distance_pct:+.2f}% from price) — that's the level they'll likely defend if price dips.")
    if bs:
        parts.append(f"The nearest heavy sell liquidity sits at {_fmt_price(bs.price)} ({_fmt_usd(bs.size_usd)} {bs.distance_pct:+.2f}% above) — main area to watch on a push higher.")
    parts.append(f"Pressure reads {report['pressure_label'].lower()}.")
    return " ".join(parts)


async def _ai_summary(report: dict, symbol: str) -> str:
    """Short trader-style paragraph from Claude. Falls back to a template if the call fails."""
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("AI_INTEGRATIONS_ANTHROPIC_API_KEY")
        if not api_key:
            return _fallback_summary(report)

        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=api_key)

        bb = report["biggest_buy"]
        bs = report["biggest_sell"]
        top_buys = report["top_buys"]
        top_sells = report["top_sells"]

        def _w_line(w: Wall) -> str:
            return f"{_fmt_price(w.price)} | {_fmt_usd(w.size_usd)} | {_fmt_dist(w.distance_pct)} | {w.confidence}"

        # Compute nearest walls + liq distances for high-leverage context
        nearest_buy = top_buys[0] if top_buys else None
        nearest_sell = top_sells[0] if top_sells else None
        nearest_buy_dist = abs(nearest_buy.distance_pct) if nearest_buy else None
        nearest_sell_dist = abs(nearest_sell.distance_pct) if nearest_sell else None

        # At 100x liquidation ≈ 1% adverse move; at 200x ≈ 0.5%
        liq_warning_lines = []
        if nearest_buy_dist is not None:
            liq_warning_lines.append(
                f"- Nearest buy wall is {nearest_buy_dist:.2f}% below price — "
                f"a SHORT here gets liquidated at 100x if price moves {1.0:.2f}% against you, "
                f"so wall sits {'INSIDE' if nearest_buy_dist < 1.0 else 'beyond'} 100x liq distance."
            )
        if nearest_sell_dist is not None:
            liq_warning_lines.append(
                f"- Nearest sell wall is {nearest_sell_dist:.2f}% above price — "
                f"a LONG here gets liquidated at 100x if price moves {1.0:.2f}% against you, "
                f"so wall sits {'INSIDE' if nearest_sell_dist < 1.0 else 'beyond'} 100x liq distance."
            )

        # Build wall behavior tags (FRESH/GROWING/SHRINKING/STABLE) per top wall
        behavior = report.get("wall_behavior") or {}
        snap_age = report.get("previous_snapshot_age_secs")

        def _w_line_with_tag(w: Wall) -> str:
            tag = ""
            if isinstance(behavior, dict):
                lbl = behavior.get(round(w.price, 8))
                if lbl:
                    tag = f" [{lbl}]"
            return _w_line(w) + tag

        # Volume + liquidation context
        vol_24h = float(report.get("volume_24h_usd", 0))
        vol_1h = float(report.get("volume_1h_usd", 0))
        liq_long = float(report.get("liq_24h_long_usd", 0))
        liq_short = float(report.get("liq_24h_short_usd", 0))

        # Wall vs 1h volume — flags walls that are massive vs flow vs trivial
        biggest_visible = max(
            (w.size_usd for w in (top_buys + top_sells)),
            default=0.0,
        )
        wall_vs_vol_note = ""
        if vol_1h > 0 and biggest_visible > 0:
            ratio = biggest_visible / vol_1h
            if ratio >= 1.0:
                wall_vs_vol_note = (
                    f"Biggest wall is {ratio:.1f}x the last hour of volume — "
                    f"genuinely heavy, will absorb meaningful flow."
                )
            elif ratio >= 0.25:
                wall_vs_vol_note = (
                    f"Biggest wall is {ratio*100:.0f}% of last hour volume — "
                    f"meaningful but can be eaten in a strong push."
                )
            else:
                wall_vs_vol_note = (
                    f"Biggest wall is only {ratio*100:.0f}% of last hour volume — "
                    f"thin vs flow, likely sweeps through."
                )

        # Broken/absorbed walls from previous snapshot
        broken = behavior.get("_broken") if isinstance(behavior, dict) else None
        broken_note = ""
        if broken:
            broken_note = "Recent wall changes:\n" + "\n".join(
                f"- {b['side'].upper()} wall at {_fmt_price(b['price'])} "
                f"({_fmt_usd(b['size_usd'])}) is GONE — broken or absorbed since last snap"
                for b in broken[:3]
            ) + "\n"

        liq_note = ""
        if liq_long > 0 or liq_short > 0:
            liq_note = (
                f"24h liquidations on this coin: longs {_fmt_usd(liq_long)}, "
                f"shorts {_fmt_usd(liq_short)} "
                f"({'longs being rekt' if liq_long > liq_short * 1.3 else 'shorts being rekt' if liq_short > liq_long * 1.3 else 'balanced'}).\n"
            )

        snap_note = ""
        if snap_age is not None:
            snap_note = f"Wall behavior tags compare against snapshot from {snap_age // 60}m {snap_age % 60}s ago.\n"

        prompt = (
            f"You are a degen crypto futures scalper running 100x-200x leverage on {symbol}.\n"
            f"At 100x: every 1% move = 100% PnL or full liquidation.\n"
            f"At 200x: every 0.5% move = 100% PnL or full liquidation.\n"
            f"Walls are where stops and liquidations cluster — price is magnetized to them.\n"
            f"FRESH walls just appeared (could be defense or spoof). GROWING means real conviction. "
            f"SHRINKING means it's getting eaten — about to break.\n\n"
            f"Current price: {_fmt_price(report['price'])}.\n"
            f"Order-book pressure: {report['pressure_label']} ({report['pressure_score']:+.2f}).\n"
            f"{snap_note}"
            f"24h volume: {_fmt_usd(vol_24h)} | last 1h: {_fmt_usd(vol_1h)}.\n"
            f"{wall_vs_vol_note}\n"
            f"{liq_note}\n"
            f"Top buy walls (price | usd | distance | confidence) [behavior]:\n"
            + "\n".join("- " + _w_line_with_tag(w) for w in top_buys[:3]) + "\n\n"
            f"Top sell walls:\n"
            + "\n".join("- " + _w_line_with_tag(w) for w in top_sells[:3]) + "\n"
            + (f"\n{broken_note}" if broken_note else "")
            + ("\nLeverage context:\n" + "\n".join(liq_warning_lines) + "\n" if liq_warning_lines else "")
            + "\n"
            "Output MUST follow this exact structure with these exact labels — no prose paragraphs, "
            "no greetings, no emojis, no markdown:\n\n"
            "BIAS: <one short sentence — which side dominates and the magnet level. "
            "Weight by FRESH/GROWING/SHRINKING tags — a SHRINKING wall in the path is NOT real defense.>\n"
            "TRADE: <LONG or SHORT> @ <entry price or tight zone>\n"
            "STOP: <price> (<distance %>)\n"
            "TP: <price> (<distance %>)\n"
            "R:R: <ratio like 1:2.5>\n"
            "LEVERAGE: <safe lev like '100x ok' or '50x max — stop too tight for 100x' or '200x ok'>\n"
            "INVALIDATION: <one price level. If price breaks this, setup is dead — cut.>\n"
            "ODDS: UP <int>% / DOWN <int>% / CHOP <int>%   (must sum to 100, be opinionated, "
            "only use 33/33/34 if data is truly indecisive)\n"
            "NOTE: <one short line — biggest risk to the trade or key signal you're weighting most>\n"
        )

        msg = await client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (msg.content[0].text or "").strip() if msg.content else ""
        return text or _fallback_summary(report)
    except Exception as e:
        logger.warning(f"AI summary failed for {symbol}: {e}")
        return _fallback_summary(report)


def _best_zone(report: dict) -> str:
    """One-line 'best zone to watch'."""
    bb = report["biggest_buy"]
    bs = report["biggest_sell"]
    if not bb and not bs:
        return "No meaningful liquidity within scan band."
    pick = bb if (bb and (not bs or bb.size_usd > bs.size_usd)) else bs
    side = "buy support" if pick.side == "buy" else "sell wall overhead"
    return (
        f"{_fmt_price(pick.price)} {side} ({_fmt_usd(pick.size_usd)}, {_fmt_dist(pick.distance_pct)}) — "
        f"if it {'breaks' if pick.side == 'sell' else 'cracks'}, "
        f"{'upside opens' if pick.side == 'sell' else 'downside opens'} fast."
    )


# ───────────────────────── Public entry point ─────────────────────────

async def scan_walls(
    user_symbol: str,
    max_band_pct: float = DEFAULT_MAX_BAND_PCT,
    min_notional_usd: float = DEFAULT_MIN_NOTIONAL_USD,
    min_distance_pct: float = DEFAULT_MIN_DISTANCE_PCT,
    top_n: int = DEFAULT_TOP_N,
    use_ai: bool = True,
) -> Optional[WallReport]:
    """Scan a coin for resting liquidity walls. Returns None if no exchange has the pair."""
    symbol = _norm_symbol(user_symbol)
    books, failed = await _fetch_all_books(symbol)
    if not books:
        return None

    mid = _mid_price(books)
    if mid <= 0:
        return None

    # Make sure min_distance < max_band, otherwise nothing matches
    if min_distance_pct >= max_band_pct:
        min_distance_pct = max_band_pct * 0.05

    buy_walls = _aggregate_walls(books, mid, "buy", max_band_pct, min_notional_usd, min_distance_pct)
    sell_walls = _aggregate_walls(books, mid, "sell", max_band_pct, min_notional_usd, min_distance_pct)

    # Build distance bands that match the user's max scan range
    bands = [(low, min(high, max_band_pct)) for low, high in DEFAULT_DISTANCE_BANDS if low < max_band_pct]
    if not bands:
        bands = [(min_distance_pct, max_band_pct)]

    # Spread top picks across distance bands so we get a ladder of zones
    top_buys = _pick_band_leaders(buy_walls, bands)[:top_n]
    top_sells = _pick_band_leaders(sell_walls, bands)[:top_n]
    pressure_label, pressure_score = _pressure(top_buys, top_sells)

    # ── Enrichment: wall behavior history, volume, liquidations (all fail-soft) ──
    from app.services import wall_intel
    prev_snapshot = wall_intel.load_previous_snapshot(symbol)
    all_current_walls = list(top_buys) + list(top_sells)
    behavior = wall_intel.classify_wall_changes(all_current_walls, prev_snapshot)
    snap_age = prev_snapshot["age_secs"] if prev_snapshot else None

    vol_ctx, liq_ctx = await asyncio.gather(
        wall_intel.get_volume_context(symbol),
        wall_intel.get_liquidation_context(symbol),
        return_exceptions=True,
    )
    if isinstance(vol_ctx, Exception):
        vol_ctx = {"volume_24h_usd": 0.0, "volume_1h_usd": 0.0}
    if isinstance(liq_ctx, Exception):
        liq_ctx = {"liq_24h_long_usd": 0.0, "liq_24h_short_usd": 0.0, "liq_24h_total_usd": 0.0}

    intermediate = {
        "price": mid,
        "biggest_buy": buy_walls[0] if buy_walls else None,
        "biggest_sell": sell_walls[0] if sell_walls else None,
        "top_buys": top_buys,
        "top_sells": top_sells,
        "pressure_label": pressure_label,
        "pressure_score": pressure_score,
        "volume_24h_usd": float(vol_ctx.get("volume_24h_usd", 0)),
        "volume_1h_usd": float(vol_ctx.get("volume_1h_usd", 0)),
        "liq_24h_long_usd": float(liq_ctx.get("liq_24h_long_usd", 0)),
        "liq_24h_short_usd": float(liq_ctx.get("liq_24h_short_usd", 0)),
        "wall_behavior": behavior,
        "previous_snapshot_age_secs": snap_age,
    }

    ai_text = await _ai_summary(intermediate, symbol) if use_ai else _fallback_summary(intermediate)

    # Persist current snapshot for the next /walls call (best-effort)
    wall_intel.save_snapshot(symbol, mid, all_current_walls)

    return WallReport(
        symbol=symbol,
        price=mid,
        biggest_buy=intermediate["biggest_buy"],
        biggest_sell=intermediate["biggest_sell"],
        top_buys=top_buys,
        top_sells=top_sells,
        pressure_label=pressure_label,
        pressure_score=pressure_score,
        exchanges_used=sorted([b["exchange"] for b in books]),
        exchanges_failed=failed,
        ai_summary=ai_text,
        best_zone_to_watch=_best_zone(intermediate),
        volume_24h_usd=intermediate["volume_24h_usd"],
        volume_1h_usd=intermediate["volume_1h_usd"],
        liq_24h_long_usd=intermediate["liq_24h_long_usd"],
        liq_24h_short_usd=intermediate["liq_24h_short_usd"],
        wall_behavior=behavior,
        previous_snapshot_age_secs=snap_age,
    )


def format_telegram(report: WallReport) -> str:
    """Render a WallReport as the HTML message the bot sends.
    All dynamic strings (symbol, AI summary) are HTML-escaped to prevent the
    Telegram parser from choking on stray <, >, or & characters."""
    p_emoji = {
        "Strongly bullish": "🟢", "Slightly bullish": "🟢",
        "Neutral": "🟡",
        "Slightly bearish": "🔴", "Strongly bearish": "🔴",
    }.get(report.pressure_label, "⚪")

    safe_symbol = html.escape(report.symbol)
    safe_summary = html.escape(report.ai_summary or "")
    safe_zone = html.escape(report.best_zone_to_watch or "")

    behavior = report.wall_behavior or {}

    def _behavior_badge(price: float) -> str:
        lbl = behavior.get(round(price, 8)) if isinstance(behavior, dict) else None
        return {
            "FRESH":     " 🆕",
            "GROWING":   " 📈",
            "SHRINKING": " 📉",
        }.get(lbl, "")

    # ── HEADER ──
    lines = [
        f"<b>💧 {safe_symbol}</b>  ·  {_fmt_price(report.price)}  ·  {p_emoji} {report.pressure_label}",
    ]

    # ── CONTEXT (compact one-liner) ──
    ctx_bits = []
    if report.volume_1h_usd > 0:
        ctx_bits.append(f"1h vol {_fmt_usd(report.volume_1h_usd)}")
    if report.volume_24h_usd > 0:
        ctx_bits.append(f"24h {_fmt_usd(report.volume_24h_usd)}")
    if report.liq_24h_long_usd > 0 or report.liq_24h_short_usd > 0:
        ctx_bits.append(
            f"liqs L {_fmt_usd(report.liq_24h_long_usd)} / S {_fmt_usd(report.liq_24h_short_usd)}"
        )
    if ctx_bits:
        lines.append(f"<i>{' · '.join(ctx_bits)}</i>")

    # ── WALLS (compact two-column-ish list) ──
    def _wall_line(w: Wall) -> str:
        tier, emoji = _size_tier(w.size_usd)
        return (
            f"  {emoji} {_fmt_price(w.price)}  "
            f"{_fmt_usd(w.size_usd)} ({_fmt_dist(w.distance_pct)})"
            f"{_behavior_badge(w.price)}"
        )

    if report.top_sells:
        lines.append("")
        lines.append("<b>🔴 SELL WALLS (resistance above)</b>")
        # show nearest-first
        for w in sorted(report.top_sells, key=lambda x: x.distance_pct):
            lines.append(_wall_line(w))

    if report.top_buys:
        lines.append("")
        lines.append("<b>🟢 BUY WALLS (support below)</b>")
        for w in sorted(report.top_buys, key=lambda x: x.distance_pct):
            lines.append(_wall_line(w))

    # ── BROKEN WALLS (only if any) ──
    broken = behavior.get("_broken") if isinstance(behavior, dict) else None
    if broken:
        lines.append("")
        lines.append("<b>💥 BROKEN / ABSORBED</b>")
        for b in broken[:3]:
            side_lbl = "buy" if b.get("side") == "buy" else "sell"
            lines.append(
                f"  · {side_lbl} {_fmt_price(b['price'])} "
                f"({_fmt_usd(b['size_usd'])}) — gone"
            )

    # ── KEY LEVELS (combined into a tight scenario block) ──
    sells_above = sorted([w for w in report.top_sells if w.price > report.price], key=lambda w: w.price)
    buys_below = sorted([w for w in report.top_buys if w.price < report.price], key=lambda w: -w.price)
    scen = []
    if sells_above:
        first = sells_above[0]
        if len(sells_above) >= 2:
            second = sells_above[1]
            move = ((second.price - first.price) / first.price) * 100
            scen.append(
                f"  📈 break {_fmt_price(first.price)} → {_fmt_price(second.price)} (+{move:.2f}%)"
            )
        else:
            scen.append(f"  📈 break {_fmt_price(first.price)} → open runway up")
    if buys_below:
        first = buys_below[0]
        if len(buys_below) >= 2:
            second = buys_below[1]
            move = ((first.price - second.price) / first.price) * 100
            scen.append(
                f"  📉 lose {_fmt_price(first.price)} → {_fmt_price(second.price)} (-{move:.2f}%)"
            )
        else:
            scen.append(f"  📉 lose {_fmt_price(first.price)} → fast drop, no support below")
    if scen:
        lines.append("")
        lines.append("<b>🔮 KEY LEVELS</b>")
        lines.extend(scen)

    # ── AI TRADE PLAN (parsed structured output) ──
    plan_block = _format_ai_plan(report.ai_summary or "")
    if plan_block:
        lines.append("")
        lines.append("<b>🧠 TRADE PLAN</b>")
        lines.append(plan_block)
    elif safe_summary:
        # Fallback: AI didn't follow the structure — show raw text in italics
        lines.append("")
        lines.append("<b>🧠 AI Read</b>")
        lines.append(f"<i>{safe_summary}</i>")

    # ── FOOTER ──
    lines.append("")
    src = ", ".join(report.exchanges_used) if report.exchanges_used else "n/a"
    if report.exchanges_failed:
        src += f" (skipped: {', '.join(report.exchanges_failed)})"
    lines.append(f"<i>Sources: {src}  ·  small &lt;$250k · 🔹 $250k-1M · 🟦 1M-5M · 🐋 5M+</i>")
    return "\n".join(lines)


# ─────────── AI plan parser: turn structured Claude output into clean HTML ───────────

_PLAN_LABELS = ("BIAS", "TRADE", "STOP", "TP", "R:R", "LEVERAGE", "INVALIDATION", "ODDS", "NOTE")
_PLAN_EMOJI = {
    "BIAS":         "🎯",
    "TRADE":        "💼",
    "STOP":         "🛑",
    "TP":           "🎁",
    "R:R":          "⚖️",
    "LEVERAGE":     "⚡",
    "INVALIDATION": "❌",
    "ODDS":         "🎲",
    "NOTE":         "📝",
}


def _format_ai_plan(raw: str) -> str:
    """Parse the structured AI output (BIAS:/TRADE:/STOP:/...) into a tidy HTML block.
    Returns '' if the AI didn't produce a recognizable structure."""
    if not raw:
        return ""
    found: dict[str, str] = {}
    # Match label at start of line (case-insensitive), allow R:R as a label
    import re as _re
    label_pattern = _re.compile(
        r"^\s*(BIAS|TRADE|STOP|TP|R\s*:\s*R|LEVERAGE|INVALIDATION|ODDS|NOTE)\s*:\s*(.+?)\s*$",
        _re.IGNORECASE,
    )
    for line in raw.splitlines():
        m = label_pattern.match(line)
        if not m:
            continue
        key = m.group(1).upper().replace(" ", "")
        if key == "R:R" or key.startswith("R:R") or key == "RR":
            key = "R:R"
        val = m.group(2).strip()
        if key in _PLAN_LABELS and val:
            found[key] = val

    if not found or "TRADE" not in found and "BIAS" not in found:
        return ""

    out_lines = []
    for k in _PLAN_LABELS:
        if k in found:
            emoji = _PLAN_EMOJI[k]
            safe_val = html.escape(found[k])
            # ODDS gets bold so it pops
            if k == "ODDS":
                out_lines.append(f"{emoji} <b>{k}:</b> <b>{safe_val}</b>")
            else:
                out_lines.append(f"{emoji} <b>{k}:</b> {safe_val}")
    return "\n".join(out_lines)
