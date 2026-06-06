"""
Gold Strategy Discovery Scanner — Claude-driven.

Finds the most profitable XAUUSD (gold) strategy by backtesting a wide roster of
candidate strategies across timeframes AND trading sessions (timezones), ranking
every (strategy × timeframe × session × risk) combination by profitability, then
asking Claude to (1) propose additional smart candidates up-front and (2) pick &
explain the single best strategy at the end.

Design notes
------------
* Reuses the existing replay engine `app/services/backtest_engine.run_backtest`
  (forex pip-mode, fees/slippage modelled) — no live API calls during replay.
* Only signals the backtest engine ACTUALLY evaluates are used as candidates.
  ICT `fx_*` signals are intentionally excluded because `eval_condition_bt`
  passes them through (no-op), which would silently inflate results.
* "Different timezones" = the four canonical FX sessions (Asian / London /
  New York / London-NY overlap). Each strategy is backtested once per timeframe,
  then its trades are bucketed by the UTC session of the entry timestamp so we
  can see which session that strategy performs best in.
* Forex is traded ~1× leverage in this app, so backtests run with leverage=1 and
  pip-based TP/SL; ranking leans on pip P&L, profit factor, win rate and
  drawdown rather than leveraged ROI.
"""
import asyncio
import json
import logging
import math
import os
import re
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

SYMBOL = "XAUUSD"
ASSET_CLASS = "forex"

# Sessions we bucket/scan. Hours mirror strategy_executor._SESSION_HOURS (UTC).
SESSIONS = ["all", "asian", "london", "new_york", "overlap"]
_SESSION_HOURS = {
    "asian":    (0, 8),
    "london":   (7, 16),
    "new_york": (13, 22),
    "overlap":  (13, 16),
}
SESSION_LABELS = {
    "all":      "All sessions (24h)",
    "asian":    "Asian (Tokyo)",
    "london":   "London",
    "new_york": "New York",
    "overlap":  "London-NY overlap",
}

# Timeframes scanned. 1h has the deepest history (~180d); 15m ~30-45d.
TIMEFRAMES = ["15m", "1h"]

# Gold pip = $0.10 (retail/broker convention). Pip-space risk variants tested per
# candidate. Each tuple is (stop_loss_pips, take_profit_pips, style, management):
#   • Targets are REALISTIC/sustainable for gold — a typical XAUUSD day ranges
#     ~$15-40 (150-400 pips), so a single-trade 500-pip ($50) target is a "blue
#     moon" event. Scalps aim 40-60 pips ($4-6); swings 120-200 pips ($12-20).
#   • management ∈ {fixed | breakeven | trail} — the scan now models stop
#     management exactly as the live executor runs it, so what's tested trades
#     the same way live (breakeven = SL→entry at halfway; trail = ratchet behind
#     price). _eval_candidate maps these to engine/exit config keys.
RISK_VARIANTS = [
    # ── Scalps (tight stop, fast, realistic 1:2 targets) ──────────────────────
    (20, 40,  "scalp", "breakeven"),   # 1:2, SL→entry at +20 pips
    (25, 50,  "scalp", "trail"),       # 1:2, trailing stop
    (30, 60,  "scalp", "fixed"),       # 1:2, plain
    # ── Swings (wider stop, sustainable target, hold the move) ────────────────
    (60, 120, "swing", "breakeven"),   # 1:2
    (80, 160, "swing", "trail"),       # 1:2
    (100, 200, "swing", "fixed"),      # 1:2
]

MIN_TRADES = 8          # a session bucket needs at least this many closed trades
MAX_CANDIDATES = 27     # hard cap on total candidates (base + Claude); sized so
                        # candidates × TIMEFRAMES × RISK_VARIANTS stays within the
                        # proven ~320-backtest budget (under the 180s inline timeout)
LEADERBOARD_SIZE = 15
CLAUDE_RESERVED_SLOTS = 8  # slots within MAX_CANDIDATES reserved for Claude's
                          # proposals so AI forex ideas aren't crowded out by the
                          # deterministic base roster (rest go to a category-diverse
                          # round-robin of the base)
_MODEL = "claude-sonnet-4-5"

# Signal vocabulary the BACKTEST ENGINE can actually evaluate (see
# backtest_engine.eval_condition_bt / _build_primary_cond). Claude is told to
# stay strictly within this set.
SUPPORTED_PRIMARY = {
    "rsi", "macd", "ema", "sma", "supertrend", "bb", "stochrsi",
    "price_momentum", "volume_spike", "breakout", "support_resistance",
    "candlestick", "divergence", "fibonacci", "fvg", "order_block",
    "market_structure", "williams_r", "adx_filter",
    # ICT forex day-trade signals — now HONESTLY evaluated by the backtest engine
    # (eval_condition_bt sync ports). Killzone/session windows let Claude build
    # genuine forex day-trade setups, not just crypto-style indicators.
    "fx_killzone", "fx_displacement", "fx_ote", "fx_cisd", "fx_sdp",
    # New ICT / Smart Money / Forex signal types
    "ifvg", "breaker_block", "mss", "choch", "liquidity_sweep",
    "mitigation_block", "supply_demand", "premium_discount", "equilibrium",
    "pin_bar", "engulfing", "inside_bar", "hh_hl", "lh_ll",
    "fib_retracement", "vwap_bounce",
}


# ── Base candidate roster (gold-appropriate, both directions) ────────────────────
def _base_roster(direction_mode: str) -> List[Dict]:
    """Diverse roster of backtest-supported strategies, expanded per direction."""
    singles = [
        ("Trend", "EMA 9/21 Cross",        "ema",          {"period": 9,  "period2": 21,  "condition": "bullish_cross"}),
        ("Trend", "EMA 20/50 Cross",       "ema",          {"period": 20, "period2": 50,  "condition": "bullish_cross"}),
        ("Trend", "SuperTrend Flip",       "supertrend",   {"condition": "bullish"}),
        ("Trend", "ADX Strong Trend",      "adx_filter",   {"condition": "trending"}),
        ("Momentum", "RSI 14 Oversold",    "rsi",          {"period": 14, "operator": "lt", "value": 30}),
        ("Momentum", "RSI Trend >55",      "rsi",          {"period": 14, "operator": "gt", "value": 55}),
        ("Momentum", "MACD Bull Cross",    "macd",         {"condition": "bullish_cross"}),
        ("Momentum", "StochRSI Oversold",  "stochrsi",     {"operator": "lt", "value": 20}),
        ("Mean Rev", "BB Lower Bounce",    "bb",           {"condition": "price_below_lower"}),
        ("Volatility", "BB Squeeze Break", "bb",           {"condition": "squeeze"}),
        ("Breakout", "20-bar Breakout",    "breakout",     {"bo_lookback": 20, "bo_pct": 0.5, "bo_dir": "up"}),
        ("Breakout", "50-bar Breakout",    "breakout",     {"bo_lookback": 50, "bo_pct": 0.3, "bo_dir": "up"}),
        ("Smart Money", "FVG Retest",      "fvg",          {"fvg_dir": "bullish", "min_gap_pct": 0.1}),
        ("Smart Money", "Order Block Retest", "order_block", {"ob_type": "bullish"}),
        ("Smart Money", "Break of Structure", "market_structure", {"condition": "bos_bullish"}),
        ("Smart Money", "Change of Character", "market_structure", {"condition": "choch_bullish"}),
        ("Divergence", "RSI Divergence",   "divergence",   {"indicator": "rsi", "direction": "bullish"}),
        ("Price Action", "Support Bounce", "support_resistance", {"condition": "at_support"}),
        # ── ICT forex day-trade signals (honestly backtested) ────────────────────
        ("ICT", "SDP Sweep+Displace",      "fx_sdp",         {"direction": "bullish", "swing_lookback": 20, "sweep_window": 5, "min_body_ratio": 2.0, "max_age": 20}),
        ("ICT", "CISD Delivery Flip",      "fx_cisd",        {"direction": "bullish", "max_run": 10}),
        ("ICT", "OTE Golden Pocket",       "fx_ote",         {"direction": "bullish", "swing_lookback": 20, "fib_low": 61.8, "fib_high": 78.6}),
        ("ICT", "Displacement Momentum",   "fx_displacement",{"direction": "bullish", "min_body_ratio": 2.5}),
        # ── New ICT / Smart Money signals ──────────────────────────────────────────
        ("ICT", "IFVG Retest",             "ifvg",           {"direction": "bullish", "min_gap_pct": 0.2}),
        ("ICT", "Breaker Block Retest",    "breaker_block",  {"direction": "bullish"}),
        ("ICT", "Market Structure Shift",  "mss",            {"direction": "bullish"}),
        ("ICT", "Change of Character",     "choch",          {"direction": "bullish"}),
        ("ICT", "Liquidity Sweep",         "liquidity_sweep",{"direction": "bullish"}),
        ("ICT", "Mitigation Block",        "mitigation_block",{"direction": "bullish"}),
        # ── Supply & Demand ────────────────────────────────────────────────────────
        ("Supply/Demand", "Supply/Demand Zone", "supply_demand",   {"direction": "bullish"}),
        ("Supply/Demand", "Premium/Discount",   "premium_discount",{"zone": "discount"}),
        ("Supply/Demand", "Equilibrium Entry",  "equilibrium",     {"direction": "bullish"}),
        # ── Price Action ───────────────────────────────────────────────────────────
        ("Price Action", "Pin Bar",           "pin_bar",      {"direction": "bullish"}),
        ("Price Action", "Engulfing Candle",  "engulfing",    {"direction": "bullish"}),
        ("Price Action", "Inside Bar Breakout","inside_bar",  {"direction": "bullish"}),
        # ── Structure ──────────────────────────────────────────────────────────────
        ("Structure", "HH/HL Bullish",        "hh_hl",        {}),
        ("Structure", "LH/LL Bearish",        "lh_ll",        {}),
        ("Structure", "Fibonacci Retracement","fib_retracement",{"direction": "bullish"}),
        ("Structure", "VWAP Bounce",          "vwap_bounce",  {"direction": "bullish"}),
    ]
    combos = [
        ("Combo", "EMA Cross + RSI>50",    "ema",  {"period": 9, "period2": 21, "condition": "bullish_cross"},
         [{"type": "rsi", "period": 14, "operator": "gt", "value": 50}]),
        ("Combo", "RSI OS + MACD Confirm", "rsi",  {"period": 14, "operator": "lt", "value": 30},
         [{"type": "macd", "condition": "bullish_cross"}]),
        ("Combo", "SuperTrend + MACD",     "supertrend", {"condition": "bullish"},
         [{"type": "macd", "condition": "bullish_cross"}]),
        ("Combo", "FVG + SuperTrend",      "fvg",  {"fvg_dir": "bullish", "min_gap_pct": 0.1},
         [{"type": "supertrend", "condition": "bullish"}]),
        ("Combo", "Order Block + RSI>50",  "order_block", {"ob_type": "bullish"},
         [{"type": "rsi", "period": 14, "operator": "gt", "value": 50}]),
        ("Combo", "Breakout + ADX Trend",  "breakout", {"bo_lookback": 20, "bo_pct": 0.5, "bo_dir": "up"},
         [{"type": "adx_filter", "condition": "trending"}]),
        # ── ICT killzone-gated forex day-trade combos ────────────────────────────
        # (SDP is already rare; gating it to a 2h killzone yields <MIN_TRADES, so it
        #  is NOT killzone-gated here — only the higher-frequency ICT signals are.)
        ("ICT", "Silver Bullet (NY KZ FVG)", "fvg", {"fvg_dir": "bullish", "min_gap_pct": 0.1},
         [{"type": "fx_killzone", "killzone": "ny_kz"}]),
        ("ICT", "NY KZ CISD",              "fx_cisd", {"direction": "bullish", "max_run": 10},
         [{"type": "fx_killzone", "killzone": "ny_kz"}]),
        ("ICT", "OTE + Displacement",      "fx_ote", {"direction": "bullish", "swing_lookback": 20, "fib_low": 61.8, "fib_high": 78.6},
         [{"type": "fx_displacement", "direction": "bullish", "min_body_ratio": 2.0}]),
        ("ICT", "CISD + Order Block",      "fx_cisd", {"direction": "bullish", "max_run": 10},
         [{"type": "order_block", "ob_type": "bullish"}]),
    ]

    out: List[Dict] = []
    for cat, label, ptype, pcfg in singles:
        out.append({"label": label, "category": cat, "primaryType": ptype,
                    "primaryCfg": dict(pcfg), "confirms": []})
    for cat, label, ptype, pcfg, conf in combos:
        out.append({"label": label, "category": cat, "primaryType": ptype,
                    "primaryCfg": dict(pcfg), "confirms": [dict(c) for c in conf]})

    return _expand_directions(out, direction_mode)


def _flip_cfg_for_short(ptype: str, cfg: Dict) -> Dict:
    """Mirror a long-biased primaryCfg into its bearish equivalent."""
    c = dict(cfg)
    if ptype == "ema" and c.get("condition") == "bullish_cross":
        c["condition"] = "bearish_cross"
    elif ptype == "macd" and c.get("condition") == "bullish_cross":
        c["condition"] = "bearish_cross"
    elif ptype == "supertrend" and c.get("condition") == "bullish":
        c["condition"] = "bearish"
    elif ptype == "rsi":
        op = c.get("operator")
        if op == "lt":   # oversold → overbought
            c["operator"], c["value"] = "gt", 100 - float(c.get("value", 30))
        elif op == "gt":
            c["operator"], c["value"] = "lt", 100 - float(c.get("value", 55))
    elif ptype == "stochrsi" and c.get("operator") == "lt":
        c["operator"], c["value"] = "gt", 100 - float(c.get("value", 20))
    elif ptype == "bb":
        if c.get("condition") == "price_below_lower":
            c["condition"] = "price_above_upper"
    elif ptype == "breakout" and c.get("bo_dir") == "up":
        c["bo_dir"] = "down"
    elif ptype == "fvg" and c.get("fvg_dir") == "bullish":
        c["fvg_dir"] = "bearish"
    elif ptype == "order_block" and c.get("ob_type") == "bullish":
        c["ob_type"] = "bearish"
    elif ptype == "market_structure":
        c["condition"] = (c.get("condition") or "").replace("bullish", "bearish")
    elif ptype == "divergence" and c.get("direction") == "bullish":
        c["direction"] = "bearish"
    elif ptype == "support_resistance" and c.get("condition") == "at_support":
        c["condition"] = "at_resistance"
    elif ptype in ("fx_sdp", "fx_cisd", "fx_ote", "fx_displacement"):
        # ICT signals carry an explicit direction; killzone is a time gate (no flip).
        if c.get("direction") == "bullish":
            c["direction"] = "bearish"
    elif ptype in ("ifvg", "breaker_block", "mss", "choch", "liquidity_sweep",
                   "mitigation_block", "supply_demand", "pin_bar", "engulfing",
                   "inside_bar", "equilibrium", "fib_retracement", "vwap_bounce"):
        if c.get("direction") == "bullish":
            c["direction"] = "bearish"
    elif ptype == "premium_discount":
        if c.get("zone") == "discount":
            c["zone"] = "premium"
    elif ptype == "hh_hl":
        pass  # hh_hl is always bullish; _expand_directions maps to lh_ll for SHORT
    elif ptype == "lh_ll":
        pass  # lh_ll is always bearish
    return c


def _flip_confirms_for_short(confirms: List[Dict]) -> List[Dict]:
    out = []
    for cf in confirms or []:
        out.append({**cf, **_flip_cfg_for_short(cf.get("type", ""), cf)})
    return out


def _expand_directions(roster: List[Dict], direction_mode: str) -> List[Dict]:
    mode = (direction_mode or "BOTH").upper()
    out: List[Dict] = []
    for r in roster:
        if mode in ("LONG", "BOTH"):
            out.append({**r, "direction": "LONG"})
        if mode in ("SHORT", "BOTH"):
            out.append({
                **r,
                "direction": "SHORT",
                "label": r["label"] + " (Short)",
                "primaryCfg": _flip_cfg_for_short(r["primaryType"], r["primaryCfg"]),
                "confirms": _flip_confirms_for_short(r.get("confirms", [])),
            })
    return out


# ── Claude: propose additional candidates (best-effort, validated) ──────────────
def _extract_json(text: str):
    """Pull the first JSON array/object out of a Claude response."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    for opener, closer in (("[", "]"), ("{", "}")):
        s = text.find(opener)
        e = text.rfind(closer)
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(text[s:e + 1])
            except Exception:
                continue
    return None


def _validate_candidate(c: Dict) -> Optional[Dict]:
    if not isinstance(c, dict):
        return None
    ptype = str(c.get("primaryType", "")).strip()
    if ptype not in SUPPORTED_PRIMARY:
        return None
    direction = str(c.get("direction", "LONG")).upper()
    if direction not in ("LONG", "SHORT"):
        direction = "LONG"
    pcfg = c.get("primaryCfg")
    if not isinstance(pcfg, dict):
        pcfg = {}
    confirms = []
    for cf in (c.get("confirms") or []):
        if isinstance(cf, dict) and str(cf.get("type", "")) in SUPPORTED_PRIMARY:
            confirms.append(cf)
    label = str(c.get("label") or ptype.upper())[:60]
    return {
        "label": label,
        "category": str(c.get("category") or "AI")[:24],
        "direction": direction,
        "primaryType": ptype,
        "primaryCfg": pcfg,
        "confirms": confirms,
        "source": "claude",
    }


async def _claude_propose_candidates(direction_mode: str, n: int = 10) -> List[Dict]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return []
    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=api_key)
        vocab = ", ".join(sorted(SUPPORTED_PRIMARY))
        prompt = (
            "You are a quantitative gold (XAUUSD) trading researcher. Propose "
            f"{n} DIVERSE candidate entry strategies to backtest on gold. Mix "
            "trend-following, momentum, mean-reversion, breakout, smart-money "
            "(FVG / order block / market structure) AND ICT forex day-trade styles "
            "(killzone-timed sweeps, displacement, OTE, CISD, SDP). Gold is traded "
            "as a forex pair, so favour genuine ICT day-trade setups, not only "
            "crypto-style indicators.\n\n"
            f"You MUST only use these primaryType values: {vocab}.\n"
            "Each confirm must also use one of those as its 'type'.\n"
            f"Direction mode: {direction_mode} (emit LONG and/or SHORT accordingly).\n\n"
            "Config key hints:\n"
            "- rsi: {period, operator(lt/gt), value}\n"
            "- ema: {period, period2, condition(bullish_cross/bearish_cross/above/below)}\n"
            "- macd: {condition(bullish_cross/bearish_cross)}\n"
            "- supertrend: {condition(bullish/bearish)}\n"
            "- bb: {condition(price_below_lower/price_above_upper/squeeze)}\n"
            "- stochrsi: {operator(lt/gt), value}\n"
            "- breakout: {bo_lookback, bo_pct, bo_dir(up/down)}\n"
            "- fvg: {fvg_dir(bullish/bearish), min_gap_pct}\n"
            "- order_block: {ob_type(bullish/bearish)}\n"
            "- market_structure: {condition(bos_bullish/bos_bearish/choch_bullish/choch_bearish)}\n"
            "- divergence: {indicator(rsi/macd), direction(bullish/bearish)}\n"
            "- adx_filter: {condition(trending)}\n"
            "- volume_spike: {multiplier}\n"
            "ICT forex day-trade signals (use these for forex/day-trade ideas):\n"
            "- fx_killzone: {killzone(london_kz/ny_kz/asian_kz/any_kz)} — time gate, "
            "best as a CONFIRM to restrict an entry to a high-probability window\n"
            "- fx_displacement: {direction(bullish/bearish/any), min_body_ratio} — "
            "institutional momentum candle\n"
            "- fx_ote: {direction(bullish/bearish), swing_lookback, fib_low, fib_high} — "
            "Optimal Trade Entry fib golden pocket (61.8-78.6%)\n"
            "- fx_cisd: {direction(bullish/bearish), max_run} — Change in State of "
            "Delivery (close back through the origin of the prior run)\n"
            "- fx_sdp: {direction(bullish/bearish), swing_lookback, sweep_window, "
            "min_body_ratio, max_age} — Sweep→Displacement→Pullback entry\n"
            "New ICT / Smart Money / Forex signals:\n"
            "- ifvg: {direction(bullish/bearish), min_gap_pct} — Inverted Fair Value Gap retest\n"
            "- breaker_block: {direction(bullish/bearish)} — Failed order block retested from other side\n"
            "- mss: {direction(bullish/bearish)} — Market Structure Shift\n"
            "- choch: {direction(bullish/bearish)} — Change of Character\n"
            "- liquidity_sweep: {direction(bullish/bearish)} — Sweep of swing high/low with reversal\n"
            "- mitigation_block: {direction(bullish/bearish)} — Retrace to origin of major move\n"
            "- supply_demand: {direction(bullish/bearish)} — Supply/Demand zone retest\n"
            "- premium_discount: {zone(discount/premium)} — Price in discount (longs) or premium (shorts)\n"
            "- equilibrium: {direction(bullish/bearish)} — 50% retracement of recent swing\n"
            "- pin_bar: {direction(bullish/bearish)} — Pin bar reversal candle\n"
            "- engulfing: {direction(bullish/bearish)} — Engulfing candle pattern\n"
            "- inside_bar: {direction(bullish/bearish)} — Inside bar breakout\n"
            "- hh_hl: {} — Higher highs / higher lows bullish structure\n"
            "- lh_ll: {} — Lower highs / lower lows bearish structure\n"
            "- fib_retracement: {direction(bullish/bearish)} — Fibonacci 0.618-0.705 entry\n"
            "- vwap_bounce: {direction(bullish/bearish)} — VWAP bounce/rejection\n\n"
            "Return ONLY a JSON array. Each item: "
            '{"label","category","direction","primaryType","primaryCfg","confirms"}. '
            "No prose."
        )
        resp = await asyncio.wait_for(
            client.messages.create(
                model=_MODEL,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=45,
        )
        text = "".join(getattr(b, "text", "") for b in resp.content)
        arr = _extract_json(text)
        if not isinstance(arr, list):
            return []
        out = []
        for c in arr:
            v = _validate_candidate(c)
            if v:
                out.append(v)
        logger.info(f"[gold-scan] Claude proposed {len(out)} valid candidates")
        return out
    except Exception as e:
        logger.warning(f"[gold-scan] Claude candidate proposal failed: {type(e).__name__}: {e}")
        return []


# ── Session bucketing + scoring ─────────────────────────────────────────────────
def _entry_session(ts_ms: int) -> List[str]:
    try:
        hour = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc).hour
    except Exception:
        return []
    return [sid for sid, (a, b) in _SESSION_HOURS.items() if a <= hour < b]


def _bucket_stats(trades: List[Dict], session: str, pip_size: float) -> Optional[Dict]:
    """Filter closed trades to a session and compute profitability stats."""
    # BREAKEVEN counts as a closed trade (for pips/pnl/drawdown) but is neither a
    # win nor a loss — win_rate is over decided (win+loss) trades only, mirroring
    # the executor's three-way label so the scanner's win% matches live.
    closed = [t for t in trades if t.get("outcome") in ("WIN", "LOSS", "BREAKEVEN")]
    if session != "all":
        closed = [t for t in closed if session in _entry_session(t.get("entry_ts", 0))]
    n = len(closed)
    if n == 0:
        return None
    wins = [t for t in closed if t["outcome"] == "WIN"]
    losses = [t for t in closed if t["outcome"] == "LOSS"]
    breakevens = [t for t in closed if t["outcome"] == "BREAKEVEN"]
    decided = len(wins) + len(losses)
    gross_w = sum(t["pnl_pct"] for t in wins)
    gross_l = abs(sum(t["pnl_pct"] for t in losses))
    pf = round(gross_w / gross_l, 2) if gross_l > 0 else (99.0 if gross_w > 0 else 0.0)
    total_pips = round(sum(float(t.get("pip_move") or 0) for t in closed), 1)

    equity, peak, max_dd = 100.0, 100.0, 0.0
    for t in closed:
        equity *= (1.0 + t["pnl_pct"] / 100.0)
        equity = max(0.0, equity)
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100)
    total_pnl = round((equity / 100.0 - 1.0) * 100.0, 2)

    return {
        "closed_trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "breakevens": len(breakevens),
        "win_rate": round(len(wins) / decided * 100, 1) if decided else 0.0,
        "profit_factor": pf,
        "total_pips": total_pips,
        "avg_pips": round(total_pips / n, 1),
        "total_pnl": total_pnl,
        "max_drawdown": round(max_dd, 2),
    }


def _score(stats: Dict) -> float:
    """Composite profitability score, credibility-weighted by trade count."""
    n = stats["closed_trades"]
    if n < MIN_TRADES:
        return -1e9
    credibility = min(1.0, n / 25.0)
    pf = min(stats["profit_factor"], 6.0)
    pnl = stats["total_pnl"]
    wr = stats["win_rate"]
    dd = stats["max_drawdown"]
    raw = (pnl * 0.6) + (pf * 9.0) + (wr * 0.25) - (dd * 0.45)
    return round(credibility * raw, 3)


# ── Backtest a single candidate across timeframes/risk; bucket by session ───────
async def _eval_candidate(cand: Dict, candle_map: Dict[str, List], days: int) -> List[Dict]:
    from app.services.backtest_engine import run_backtest
    from app.services.forex_engine import pip_size as _pip_size_fn

    results: List[Dict] = []
    pip_sz = _pip_size_fn(SYMBOL)
    for tf in TIMEFRAMES:
        candles = candle_map.get(tf)
        if not candles or len(candles) < 120:
            continue
        # Reference close to convert pip distances → % of price for trailing.
        try:
            ref_close = float(candles[-1][4]) or 2000.0
        except Exception:
            ref_close = 2000.0
        for sl_pips, tp_pips, style, mgmt in RISK_VARIANTS:
            # Map management style → engine/exit config keys (same vocab the live
            # executor + AI compiler use, so test == live).
            be_at_pct = 50 if mgmt == "breakeven" else 0
            trail_on = (mgmt == "trail")
            trail_pct = round((sl_pips * pip_sz / ref_close) * 100.0, 4) if trail_on else 0.0
            cfg = {
                "direction": cand["direction"],
                "primaryType": cand["primaryType"],
                "primaryCfg": cand["primaryCfg"],
                "confirms": cand.get("confirms", []),
                "timeframe": tf,
                "singleCoin": SYMBOL,
                "asset_class": ASSET_CLASS,
                "leverage": 1,
                "take_profit_pips": tp_pips,
                "stop_loss_pips": sl_pips,
                "maxHoldHours": 72,
                "breakeven_at_pct": be_at_pct,
                "trailing_stop": trail_on,
                "trailing_stop_pct": trail_pct,
            }
            try:
                res = await run_backtest(cfg, days, precomputed_candles=candles,
                                         precomputed_source_label=f"{SYMBOL} {tf}")
            except Exception as e:
                logger.debug(f"[gold-scan] backtest err {cand['label']} {tf}: {e}")
                continue
            if res.get("error"):
                continue
            trades = res.get("trades", [])
            rr = round(tp_pips / sl_pips, 2) if sl_pips else 0
            for session in SESSIONS:
                st = _bucket_stats(trades, session, pip_sz)
                if not st or st["closed_trades"] < MIN_TRADES:
                    continue
                results.append({
                    "label": cand["label"],
                    "category": cand["category"],
                    "direction": cand["direction"],
                    "primaryType": cand["primaryType"],
                    "primaryCfg": cand["primaryCfg"],
                    "confirms": cand.get("confirms", []),
                    "source": cand.get("source", "base"),
                    "timeframe": tf,
                    "session": session,
                    "session_label": SESSION_LABELS[session],
                    "sl_pips": sl_pips,
                    "tp_pips": tp_pips,
                    "rr": rr,
                    "style": style,
                    "management": mgmt,
                    "breakeven_at_pct": be_at_pct,
                    "trailing_stop": trail_on,
                    "trailing_stop_pct": trail_pct,
                    "stats": st,
                    "score": _score(st),
                })
    return results


# ── Claude: synthesize / pick the best from the leaderboard ─────────────────────
def _build_prompt_for(entry: Dict) -> str:
    """Natural-language description the user can drop into the AI builder."""
    sess = "" if entry["session"] == "all" else f" only during the {SESSION_LABELS[entry['session']]} session"
    parts = [f"{entry['primaryType']} {json.dumps(entry['primaryCfg'])}"]
    for cf in entry.get("confirms", []):
        parts.append(f"confirmed by {cf.get('type')} {json.dumps({k: v for k, v in cf.items() if k != 'type'})}")
    sig = " AND ".join(parts)
    mgmt = entry.get("management", "fixed")
    if mgmt == "breakeven":
        mgmt_txt = (f" Move the stop loss to breakeven once the trade is "
                    f"{entry.get('breakeven_at_pct', 50)}% of the way to target.")
    elif mgmt == "trail":
        mgmt_txt = (f" Use a trailing stop of {entry.get('trailing_stop_pct', 0)}% "
                    f"behind price to lock in profit.")
    else:
        mgmt_txt = ""
    return (
        f"Build a {entry['direction']} gold (XAUUSD) strategy on the {entry['timeframe']} "
        f"timeframe that enters when {sig}{sess}. Use a stop loss of {entry['sl_pips']} pips "
        f"and take profit of {entry['tp_pips']} pips.{mgmt_txt}"
    )


def _build_name_for(entry: Dict) -> str:
    """Short saved-strategy name for an entry (used by 'Build all')."""
    sess = "" if entry["session"] == "all" else f" {entry['session_label'].split(' ')[0]}"
    style = (entry.get("style") or "").title()
    mgmt = entry.get("management", "fixed")
    mgmt_tag = {"breakeven": "BE", "trail": "Trail"}.get(mgmt, "")
    name = f"Gold {entry['label']} {entry['direction']} {entry['timeframe']}{sess}"
    suffix = " ".join(p for p in (style, mgmt_tag) if p)
    if suffix:
        name = f"{name} ({suffix})"
    return name[:60]


async def _claude_pick_best(leaderboard: List[Dict], days: int) -> Optional[Dict]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not leaderboard:
        return None
    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=api_key)
        rows = []
        for i, e in enumerate(leaderboard):
            s = e["stats"]
            rows.append(
                f"[{i}] {e['label']} | {e['direction']} | {e['timeframe']} | "
                f"{e['session_label']} | SL {e['sl_pips']}/TP {e['tp_pips']} (RR {e['rr']}) | "
                f"mgmt={e.get('management', 'fixed')} | "
                f"trades={s['closed_trades']} winrate={s['win_rate']}% PF={s['profit_factor']} "
                f"pips={s['total_pips']} pnl={s['total_pnl']}% maxDD={s['max_drawdown']}%"
            )
        prompt = (
            f"These are the top backtested gold (XAUUSD) strategies over the last {days} days, "
            "ranked by a composite profitability score. Pick the SINGLE best one to trade live, "
            "balancing profit factor, win rate, total pips, drawdown AND sample size "
            "(more trades = more reliable; be wary of high returns from few trades).\n\n"
            + "\n".join(rows) +
            '\n\nReturn ONLY JSON: {"best_index": <int>, "strategy_name": "<short catchy name>", '
            '"rationale": "<2-4 sentences explaining why this is the best and what session/timeframe edge it exploits>"}'
        )
        resp = await asyncio.wait_for(
            client.messages.create(model=_MODEL, max_tokens=600,
                                   messages=[{"role": "user", "content": prompt}]),
            timeout=45,
        )
        text = "".join(getattr(b, "text", "") for b in resp.content)
        obj = _extract_json(text)
        if not isinstance(obj, dict):
            return None
        idx = int(obj.get("best_index", 0))
        if not (0 <= idx < len(leaderboard)):
            idx = 0
        return {
            "index": idx,
            "strategy_name": str(obj.get("strategy_name") or leaderboard[idx]["label"])[:60],
            "rationale": str(obj.get("rationale") or "")[:800],
        }
    except Exception as e:
        logger.warning(f"[gold-scan] Claude pick failed: {type(e).__name__}: {e}")
        return None


# ── Public entry point ──────────────────────────────────────────────────────────
async def run_gold_discovery(days: int = 90, direction_mode: str = "BOTH",
                             progress_cb: Optional[Callable[[str], None]] = None) -> Dict:
    """
    Run the full Claude-driven gold strategy discovery scan.

    Returns:
        { ok, symbol, days, timeframes, sessions, candidates_tested,
          combos_evaluated, leaderboard:[...], best:{...}, generated_by }
    """
    def _progress(msg: str):
        if progress_cb:
            try: progress_cb(msg)
            except Exception: pass
        logger.info(f"[gold-scan] {msg}")

    if days not in (30, 90, 180):
        days = 90
    direction_mode = (direction_mode or "BOTH").upper()
    if direction_mode not in ("LONG", "SHORT", "BOTH"):
        direction_mode = "BOTH"

    # 1) Fetch candles once per timeframe.
    _progress("Fetching gold history…")
    from app.services.tradfi_prices import get_klines
    candle_map: Dict[str, List] = {}
    coverage: Dict[str, float] = {}   # actual days of history fetched per timeframe
    for tf in TIMEFRAMES:
        per_day = {"15m": 96, "1h": 24}.get(tf, 24)
        # Request more than the window so the provider returns its full depth;
        # the source itself caps how far back gold data goes (15m ≈ 1 month,
        # 1h ≈ several months), so we report the ACTUAL coverage rather than
        # pretending every window is fully covered.
        limit = min(per_day * days + 120, 6000)
        try:
            ks = await get_klines(SYMBOL, ASSET_CLASS, tf, limit)
        except Exception as e:
            logger.warning(f"[gold-scan] candle fetch {tf} failed: {e}")
            ks = []
        n = len(ks) if ks else 0
        if ks and n >= 120:
            candle_map[tf] = ks
            try:
                coverage[tf] = round((int(ks[-1][0]) - int(ks[0][0])) / 86400000.0, 1)
            except Exception:
                coverage[tf] = 0.0
        else:
            logger.warning(f"[gold-scan] {tf}: got {n} bars (need ≥120)")
    if not candle_map:
        return {
            "ok": False,
            "error": (
                "Could not fetch gold (XAUUSD) historical data. "
                "Ensure FMP_API_KEY is set (stable intraday plan) or connect cTrader."
            ),
        }

    # 2) Build candidate roster (base + Claude proposals).
    _progress("Asking Claude to propose strategies…")
    candidates = _base_roster(direction_mode)
    claude_cands = await _claude_propose_candidates(direction_mode, n=12)

    def _sig(c: Dict) -> tuple:
        return (c["primaryType"], c["direction"],
                json.dumps(c.get("primaryCfg", {}), sort_keys=True),
                json.dumps(c.get("confirms", []), sort_keys=True))

    # The base roster (after LONG/SHORT expansion) is far larger than MAX_CANDIDATES,
    # so a naive base-order [:N] cut keeps only the first few categories' singles and
    # silently drops every later category (notably the ICT/forex ideas) AND all of
    # Claude's proposals. Instead: round-robin the base by category so the cap keeps a
    # spread across categories (trend, momentum, smart-money, ICT, combo, …), and
    # RESERVE a block of slots for Claude's proposals so AI forex ideas aren't crowded
    # out by the deterministic base roster.
    def _round_robin_by_category(cands: List[Dict]) -> List[Dict]:
        from collections import OrderedDict
        buckets: "OrderedDict[str, List[Dict]]" = OrderedDict()
        for c in cands:
            buckets.setdefault(c.get("category", "?"), []).append(c)
        queues = list(buckets.values())
        out: List[Dict] = []
        while any(queues):
            for q in queues:
                if q:
                    out.append(q.pop(0))
        return out

    seen: set = set()
    merged: List[Dict] = []
    claude_keep = claude_cands[:CLAUDE_RESERVED_SLOTS]
    base_budget = max(0, MAX_CANDIDATES - len(claude_keep))

    for c in _round_robin_by_category(candidates):
        if len(merged) >= base_budget:
            break
        sig = _sig(c)
        if sig in seen:
            continue
        seen.add(sig)
        merged.append(c)

    for c in claude_keep:
        if len(merged) >= MAX_CANDIDATES:
            break
        sig = _sig(c)
        if sig in seen:
            continue
        seen.add(sig)
        merged.append(c)

    # 3) Backtest every candidate × timeframe × risk, bucket by session.
    _progress(f"Backtesting {len(merged)} strategies across {len(candle_map)} timeframes & {len(SESSIONS)} sessions…")
    all_results: List[Dict] = []
    for cand in merged:
        all_results.extend(await _eval_candidate(cand, candle_map, days))

    if not all_results:
        return {"ok": False, "error": "No strategy produced enough trades on gold to rank. Try a longer window."}

    # 4) Rank. Keep best variant per (label, direction) so the board isn't
    #    dominated by RR/session permutations of one idea.
    all_results.sort(key=lambda r: r["score"], reverse=True)

    # Keep only the single best (timeframe × session × risk) variant per strategy
    # IDEA so the board surfaces diverse ideas rather than permutations of one.
    def _dedupe_by_idea(rows: List[Dict]) -> List[Dict]:
        seen, out = set(), []
        for r in rows:
            k = (r["label"], r["direction"])
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
        return out

    # Swings out-score scalps on raw pips, so a single score-ranked board hides
    # every scalp. Reserve roughly half the board for scalps, then fill the rest
    # with swings — but each strategy IDEA (label, direction) may appear ONCE on
    # the whole board (a global `used_ideas` set), so an idea shows EITHER its
    # best scalp OR its best swing profile, never both.
    scalp_best = _dedupe_by_idea([r for r in all_results if r.get("style") == "scalp"])
    swing_best = _dedupe_by_idea([r for r in all_results if r.get("style") == "swing"])
    half = LEADERBOARD_SIZE // 2
    leaderboard: List[Dict] = []
    used_ideas: set = set()

    def _take(rows: List[Dict], cap: int) -> None:
        for r in rows:
            if len(leaderboard) >= cap:
                break
            idea = (r["label"], r["direction"])
            if idea in used_ideas:
                continue
            used_ideas.add(idea)
            leaderboard.append(r)

    _take(scalp_best, half)                 # ~half the board reserved for scalps
    _take(swing_best, LEADERBOARD_SIZE)     # fill remainder with swing ideas
    _take(scalp_best, LEADERBOARD_SIZE)     # backfill leftover scalps if swings ran out
    leaderboard.sort(key=lambda r: r["score"], reverse=True)

    # Attach a build name + NL build prompt to EVERY leaderboard entry so the UI
    # can build any single one (or all of them at once via "Build all").
    for e in leaderboard:
        e["build_prompt"] = _build_prompt_for(e)
        e["build_name"] = _build_name_for(e)

    # 5) Claude picks & explains the winner.
    _progress("Asking Claude to pick the best…")
    pick = await _claude_pick_best(leaderboard, days)
    best_idx = pick["index"] if pick else 0
    best_entry = dict(leaderboard[best_idx])
    best_entry["strategy_name"] = pick["strategy_name"] if pick else best_entry["label"]
    best_entry["rationale"] = pick["rationale"] if pick else (
        "Top-ranked by composite profitability score (profit factor, win rate, "
        "total pips and drawdown, weighted by sample size)."
    )
    best_entry["build_prompt"] = _build_prompt_for(best_entry)

    return {
        "ok": True,
        "symbol": SYMBOL,
        "days": days,
        "direction_mode": direction_mode,
        "timeframes": list(candle_map.keys()),
        "coverage_days": coverage,
        "sessions": SESSIONS,
        "candidates_tested": len(merged),
        "combos_evaluated": len(all_results),
        "leaderboard": leaderboard,
        "best": best_entry,
        "generated_by": "claude" if pick else "ranker",
        "ai_proposed": len(claude_cands),
    }
