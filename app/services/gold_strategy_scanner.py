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
# candidate. Each tuple is (stop_loss_pips, take_profit_pips, style) → RR derived.
# We test BOTH scalp (tight stop, ~100-pip target, in & out fast) and swing (wide
# stop, big target) profiles so the scanner surfaces fast scalps as well as runners.
RISK_VARIANTS = [
    # ── Scalps (tight, fast, ~100-pip targets; gold pip=$0.10 → 100 pips=$10) ──
    (50, 100, "scalp"),    # 1:2
    (60, 150, "scalp"),    # 1:2.5
    (40, 120, "scalp"),    # 1:3
    # ── Swings (wide stop, big target, hold for the move) ─────────────────────
    (200, 400, "swing"),   # 1:2
    (250, 500, "swing"),   # 1:2
    (150, 450, "swing"),   # 1:3
]

MIN_TRADES = 8          # a session bucket needs at least this many closed trades
MAX_CANDIDATES = 27     # hard cap on total candidates (base + Claude); sized so
                        # candidates × TIMEFRAMES × RISK_VARIANTS stays within the
                        # proven ~320-backtest budget (under the 180s inline timeout)
LEADERBOARD_SIZE = 15
_MODEL = "claude-sonnet-4-5"

# Signal vocabulary the BACKTEST ENGINE can actually evaluate (see
# backtest_engine.eval_condition_bt / _build_primary_cond). Claude is told to
# stay strictly within this set.
SUPPORTED_PRIMARY = {
    "rsi", "macd", "ema", "sma", "supertrend", "bb", "stochrsi",
    "price_momentum", "volume_spike", "breakout", "support_resistance",
    "candlestick", "divergence", "fibonacci", "fvg", "order_block",
    "market_structure", "williams_r", "adx_filter",
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
            "trend-following, momentum, mean-reversion, breakout and smart-money "
            "(FVG / order block / market structure) styles.\n\n"
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
            "- volume_spike: {multiplier}\n\n"
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
    closed = [t for t in trades if t.get("outcome") in ("WIN", "LOSS")]
    if session != "all":
        closed = [t for t in closed if session in _entry_session(t.get("entry_ts", 0))]
    n = len(closed)
    if n == 0:
        return None
    wins = [t for t in closed if t["outcome"] == "WIN"]
    losses = [t for t in closed if t["outcome"] == "LOSS"]
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
        "win_rate": round(len(wins) / n * 100, 1),
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
        for sl_pips, tp_pips, style in RISK_VARIANTS:
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
    return (
        f"Build a {entry['direction']} gold (XAUUSD) strategy on the {entry['timeframe']} "
        f"timeframe that enters when {sig}{sess}. Use a stop loss of {entry['sl_pips']} pips "
        f"and take profit of {entry['tp_pips']} pips."
    )


def _build_name_for(entry: Dict) -> str:
    """Short saved-strategy name for an entry (used by 'Build all')."""
    sess = "" if entry["session"] == "all" else f" {entry['session_label'].split(' ')[0]}"
    style = (entry.get("style") or "").title()
    name = f"Gold {entry['label']} {entry['direction']} {entry['timeframe']}{sess}"
    if style:
        name = f"{name} ({style})"
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
        if ks and len(ks) >= 120:
            candle_map[tf] = ks
            try:
                coverage[tf] = round((int(ks[-1][0]) - int(ks[0][0])) / 86400000.0, 1)
            except Exception:
                coverage[tf] = 0.0
    if not candle_map:
        return {"ok": False, "error": "Could not fetch gold (XAUUSD) historical data. Try again shortly."}

    # 2) Build candidate roster (base + Claude proposals).
    _progress("Asking Claude to propose strategies…")
    candidates = _base_roster(direction_mode)
    claude_cands = await _claude_propose_candidates(direction_mode, n=12)
    # Dedupe by (primaryType, direction, sorted primaryCfg) signature.
    seen = set()
    merged: List[Dict] = []
    for c in candidates + claude_cands:
        sig = (c["primaryType"], c["direction"],
               json.dumps(c.get("primaryCfg", {}), sort_keys=True),
               json.dumps(c.get("confirms", []), sort_keys=True))
        if sig in seen:
            continue
        seen.add(sig)
        merged.append(c)
    merged = merged[:MAX_CANDIDATES]

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
